import json
import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances, silhouette_score

st.set_page_config(page_title="Dashboard Analisis AI Kreatif", layout="wide")

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 110})

# ==========================================================
# KONFIGURASI (disamakan dengan notebook)
# ==========================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'datas'


def cari_berkas(nama):
    """Cari di datas/ dulu, lalu di folder app.py."""
    for p in (DATA_DIR / nama, BASE_DIR / nama):
        if p.exists():
            return p
    return DATA_DIR / nama      # tidak ada: pesan error memakai path ini


PATH_DATA = cari_berkas('dataset_hasil.csv')
PATH_RINGKASAN = cari_berkas('ringkasan_cluster.csv')
PATH_EVAL = cari_berkas('evaluasi_sentimen.csv')
PATH_STOPWORDS = cari_berkas('stopwords.txt')
PATH_TAHAP = cari_berkas('jumlah_data_per_tahap.csv')
PATH_MODEL = cari_berkas('model_regresi.joblib')
PATH_PARAM = cari_berkas('parameter_model.json')
PATH_UJI = cari_berkas('prediksi_data_uji.csv')

INFO_PENGAMBILAN = {
    'Sumber': 'Platform X, dikumpulkan dengan tweet-harvest (Google Colab)',
    'Kata kunci': 'ilustrator ai, hak cipta ai, seniman ai, penulis ai, desainer ai, musisi ai',
    'Penyaringan periode': '2025-01-01 s.d. 2026-10-11 (kolom created_at)',
    'Waktu pengambilan': 'ISI sesuai catatan crawling kelompok (sertakan zona waktu)',   # ✏️ wajib diisi
}

KOLOM_WAJIB = ['text', 'created_at', 'like_count']
URUTAN_SENTIMEN = ['positif', 'netral', 'negatif']
WARNA_SENTIMEN = {'positif': '#2a78d6', 'netral': '#a3a29c', 'negatif': '#e34948'}
WARNA_CLUSTER = ['#2a78d6', '#eb6834', '#1baf7a']
NAMA_TEMA = {
    0: 'Hak Cipta, Regulasi, dan Hukum terkait AI',
    1: 'Dampak AI bagi Seniman, Musisi, dan Desainer: Kritik dan Kekhawatiran',
    2: 'Tuduhan dan Deteksi Karya Buatan AI pada Penulis dan Ilustrator',
}
NAMA_FITUR = {
    'like_count': 'Jumlah likes', 'retweet_count': 'Jumlah retweet', 'reply_count': 'Jumlah reply',
    'jumlah_kata': 'Jumlah kata', 'jumlah_hashtag': 'Jumlah hashtag', 'panjang_karakter': 'Panjang karakter',
    'jumlah_mention': 'Jumlah mention', 'ada_url': 'Ada tautan (0/1)', 'balasan': 'Postingan balasan (0/1)',
    'usia_hari': 'Usia postingan (hari)', 'sent_polaritas': 'Skor sentimen (-1 s.d. 1)',
}
RANDOM_STATE = 42

STOPWORDS_TAMBAHAN = [
    # kata topik (muncul di hampir semua postingan)
    'ai', 'artificial', 'intelligence', 'kecerdasan', 'buatan',
    # kata sambung umum
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'karena', 'juga', 'tapi',
    'kalau', 'akan', 'atau', 'tidak', 'bisa', 'ada', 'jadi', 'sama', 'buat', 'pakai', 'gara',
    # singkatan & bahasa gaul
    'yg', 'ga', 'gak', 'nggak', 'ngga', 'enggak', 'gk', 'tdk', 'aja', 'udah', 'udh', 'dah',
    'sih', 'deh', 'dong', 'kok', 'nih', 'tuh', 'lho', 'loh', 'wkwk', 'wkwkwk', 'haha', 'hahaha',
    'kalo', 'klo', 'tp', 'dgn', 'utk', 'bgt', 'banget', 'jg', 'gw', 'gue', 'gua', 'lu', 'lo', 'elo',
    'km', 'nya', 'emang', 'gitu', 'gini', 'kayak', 'kaya', 'kek', 'pake', 'bikin', 'org', 'orang',
    # kata bahasa Inggris yang umum
    'the', 'and', 'to', 'of', 'is', 'in', 'it', 'for', 'this', 'that', 'you', 'with', 'are', 'not',
]
# kata kunci crawling: hanya dibuang dari word cloud (clustering memakai STOPWORDS saja)
KATA_KUNCI_WORDCLOUD = ['hak', 'cipta', 'desainer', 'ilustrator', 'seniman', 'penulis', 'musisi']


# ==========================================================
# FUNGSI BANTU
# ==========================================================
@st.cache_data(show_spinner=False)
def muat_stopwords():
    """Utamakan stopwords.txt dari notebook (hasil identik, tanpa internet); NLTK sebagai cadangan."""
    dasar = []
    if PATH_STOPWORDS.exists():
        dasar = PATH_STOPWORDS.read_text(encoding='utf-8').split()
    else:
        try:
            nltk.download('stopwords', quiet=True)
            dasar = stopwords.words('indonesian')
        except Exception:
            pass
    return sorted(set(dasar) | set(STOPWORDS_TAMBAHAN)), len(dasar) > 0


@st.cache_data(show_spinner=False)
def baca_csv(isi: bytes) -> pd.DataFrame:
    """Baca CSV (tahan BOM dan pemisah ';' atau ',')."""
    import io
    awal = isi[:5000].decode('utf-8-sig', errors='ignore')
    header = awal.splitlines()[0] if awal.strip() else ''
    sep = ';' if header.count(';') > header.count(',') else ','
    return pd.read_csv(io.BytesIO(isi), sep=sep, encoding='utf-8-sig')


def baca_opsional(path: Path):
    if not path.exists():
        return None
    try:
        return baca_csv(path.read_bytes())
    except Exception as e:
        st.warning(f"Berkas {path.name} tidak dapat dibaca: {e}")
        return None


def ringkas_evaluasi():
    ev = baca_opsional(PATH_EVAL)
    if ev is None:
        return None
    ev.columns = ev.columns.str.strip().str.lower()
    if not {'label_manual', 'prediksi_model'}.issubset(ev.columns):
        return None
    a = ev['label_manual'].astype(str).str.strip().str.lower()
    b = ev['prediksi_model'].astype(str).str.strip().str.lower()
    n = len(ev)
    if n == 0:
        return None
    vc = a.value_counts()
    return {'n': n, 'k': int((a == b).sum()), 'kelas': vc.index[0], 'dasar': vc.iloc[0] / n * 100}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    pen = 1 + z ** 2 / n
    pusat = (p + z ** 2 / (2 * n)) / pen
    sel = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / pen
    return max(0.0, pusat - sel), min(1.0, pusat + sel)


def cmap_gelap(nama):
    # pakai setengah bagian gelap colormap agar kata terbaca
    warna = plt.get_cmap(nama)(np.linspace(0.5, 1, 256))
    return LinearSegmentedColormap.from_list(nama, warna)


def siapkan_sentimen(df):
    """Mengembalikan (data dengan sentimen valid, jumlah baris dikeluarkan). None bila kolom tidak ada."""
    if 'sentimen' not in df.columns:
        return None, 0
    s = df['sentimen'].astype(str).str.strip().str.lower()
    valid = s.isin(URUTAN_SENTIMEN)
    d = df.loc[valid].copy()
    d['sentimen'] = s[valid]
    return d, int((~valid).sum())


def siapkan_cluster(df):
    """Mengembalikan (data dengan cluster valid, jumlah baris dikeluarkan). None bila kolom tidak ada."""
    if 'cluster' not in df.columns:
        return None, 0
    c = pd.to_numeric(df['cluster'], errors='coerce')
    valid = c.notna()
    d = df.loc[valid].copy()
    d['cluster'] = c[valid].astype(int)
    peta = d['cluster'].map(NAMA_TEMA)
    if 'tema' in d.columns:
        d['tema'] = d['tema'].where(d['tema'].notna(), peta)
    else:
        d['tema'] = peta
    d['tema'] = d['tema'].fillna('Cluster ' + d['cluster'].astype(str))
    return d, int((~valid).sum())


def kesesuaian_query(dc):
    """Rata-rata persen postingan tiap kata kunci crawling yang jatuh ke satu cluster yang sama."""
    if 'query' not in dc.columns or dc['query'].nunique() < 2:
        return None
    ct = pd.crosstab(dc['query'], dc['cluster'], normalize='index')
    return float(ct.max(axis=1).mean() * 100)


def baris_regresi(df):
    """Rincian baris data yang valid untuk regresi (likes valid dan teks tidak kosong)."""
    like = pd.to_numeric(df['like_count'], errors='coerce')
    tidak_valid = like.isna() | (like < 0)
    kosong = df['text'].fillna('').astype(str).str.strip() == ''
    return {'dipakai': int((~tidak_valid & ~kosong).sum()),
            'like_tidak_valid': int(tidak_valid.sum()),
            'teks_kosong': int((kosong & ~tidak_valid).sum())}


# ==========================================================
# ANALISIS (di-cache)
# ==========================================================
@st.cache_data(show_spinner='Menghitung representasi TF-IDF...')
def analisis_cluster(teks: tuple, label: tuple, stop_cluster: tuple):
    """Representasi TF-IDF sama dengan notebook (stopwords, min_df=2, max_df=0.9)."""
    from scipy.sparse import csr_matrix
    vec = TfidfVectorizer(stop_words=list(stop_cluster), token_pattern=r'(?u)\b[a-z]{3,}\b',
                          min_df=2, max_df=0.9)
    X = vec.fit_transform(pd.Series(list(teks)).fillna(''))
    label = np.array(label)
    kata = vec.get_feature_names_out()
    koordinat = TruncatedSVD(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    ada_kata = X.getnnz(axis=1) > 0

    per_cluster = {}
    for c in sorted(set(label.tolist())):
        m = label == c
        pusat = np.asarray(X[m].mean(axis=0)).ravel()
        urut = pusat.argsort()[::-1]
        idx = np.where(m & ada_kata)[0]
        jarak = pairwise_distances(X[idx], csr_matrix(pusat)).ravel()
        per_cluster[int(c)] = {
            'top5': [kata[i] for i in urut[:5]],
            'top10': [kata[i] for i in urut[:10]],
            'contoh': idx[np.argsort(jarak)[:3]].tolist(),
        }
    sil = silhouette_score(X, label, metric='cosine') if len(set(label.tolist())) > 1 else float('nan')
    return koordinat, per_cluster, float(sil), int((~ada_kata).sum())


@st.cache_data(show_spinner='Membuat word cloud...')
def buat_wordcloud(teks: tuple, sentimen: tuple, stop: tuple):
    from wordcloud import WordCloud
    stopset = set(stop)
    peta_warna = {'negatif': 'Reds', 'netral': 'Greys', 'positif': 'Greens'}
    hasil = {}
    for s in ['negatif', 'netral', 'positif']:
        kata = ' '.join(t for t, ss in zip(teks, sentimen) if ss == s).split()
        kata = [w for w in kata if w not in stopset and len(w) >= 3]
        if kata:
            wc = WordCloud(width=600, height=400, background_color='white', colormap=cmap_gelap(peta_warna[s]))
            hasil[s] = wc.generate_from_frequencies(Counter(kata)).to_array()
        else:
            hasil[s] = None
    return hasil


@st.cache_resource(show_spinner=False)
def muat_regresi():
    """Memuat hasil regresi dari notebook.
    Prioritas: model_regresi.joblib. Cadangan: parameter_model.json (+ prediksi_data_uji.csv).
    Hanya muat berkas buatan sendiri (joblib/pickle dapat menjalankan kode)."""
    hasil = {'p': None, 'objek': {}, 'uji': None, 'versi_model': None, 'catatan': []}
    kunci = ('meta', 'seleksi_fitur', 'model_utama', 'pembanding_dasar', 'model_ketentuan')

    if PATH_MODEL.exists():
        try:
            bundel = joblib.load(PATH_MODEL)
            hasil['p'] = {k: bundel[k] for k in kunci if k in bundel}
            hasil['objek'] = bundel.get('objek_model') or {}
            hasil['versi_model'] = bundel.get('versi_sklearn')
            if bundel.get('uji') is not None:
                hasil['uji'] = pd.DataFrame(bundel['uji'])
        except Exception as e:
            hasil['catatan'].append(f"{PATH_MODEL.name} tidak dapat dimuat ({type(e).__name__}); "
                                    "memakai parameter_model.json bila tersedia.")
    if hasil['p'] is None and PATH_PARAM.exists():
        try:
            hasil['p'] = json.loads(PATH_PARAM.read_text(encoding='utf-8'))
        except Exception as e:
            hasil['catatan'].append(f"{PATH_PARAM.name} tidak dapat dibaca ({type(e).__name__}).")
    if hasil['uji'] is None and PATH_UJI.exists():
        try:
            hasil['uji'] = pd.read_csv(PATH_UJI)
        except Exception:
            pass
    wajib = {'meta', 'seleksi_fitur', 'model_utama', 'pembanding_dasar'}
    if hasil['p'] is not None and not wajib.issubset(hasil['p']):
        hasil['catatan'].append("Berkas hasil regresi tidak lengkap; jalankan ulang sel ekspor di notebook.")
        hasil['p'] = None
    return hasil


def estimasi(blok, nilai, model=None):
    """Pakai objek model bila ada; jika tidak, hitung dari intersep + koefisien (hasilnya sama)."""
    if model is not None:
        x = pd.DataFrame([[nilai[f] for f in blok['fitur']]], columns=blok['fitur'])
        return float(model.predict(x)[0])
    return float(blok['intercept'] + sum(c * nilai[f] for f, c in zip(blok['fitur'], blok['koefisien'])))


def gambar_heatmap_param(sel):
    label, mat = sel['label_heatmap'], np.array(sel['matriks_heatmap'])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    img = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
    n = len(label)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label, rotation=40, ha='right')
    ax.set_yticklabels(label)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(mat[i, j]) > 0.6 else '#2c2c2a')
    ax.grid(False)
    ax.set_title('Korelasi linear (Pearson), data latih', fontsize=11)
    fig.colorbar(img, ax=ax, shrink=0.8, label='Koefisien korelasi Pearson')
    plt.tight_layout()
    return fig


def form_estimasi(blok, model, kunci):
    """Formulir estimasi likes untuk satu model (fitur mengikuti blok['fitur'])."""
    with st.form(f"form_{kunci}"):
        nilai = {}
        for kol, f, lab in zip(st.columns(len(blok['fitur'])), blok['fitur'], blok['label_fitur']):
            rg = blok['rentang'][f]
            if all(float(rg[x]).is_integer() for x in ('min', 'maks', 'median')):
                nilai[f] = kol.number_input(lab, min_value=int(rg['min']), value=int(rg['median']),
                                            step=1, key=f"{kunci}_{f}")
            else:
                nilai[f] = kol.number_input(lab, min_value=float(rg['min']), value=float(rg['median']),
                                            step=0.01, key=f"{kunci}_{f}")
        kirim = st.form_submit_button("Hitung estimasi")
    if kirim:
        est = estimasi(blok, nilai, model)
        st.metric("Estimasi likes (nilai mentah model)", f"{est:,.1f}")
        if est < 0:
            st.warning("Estimasi negatif tidak bermakna sebagai jumlah likes. Ini menunjukkan keterbatasan model linear.")
        luar = [lab for f, lab in zip(blok['fitur'], blok['label_fitur']) if nilai[f] > blok['rentang'][f]['maks']]
        if luar:
            st.warning("Masukan di luar rentang data untuk: " + ", ".join(luar) + ". Estimasi tidak andal.")
        st.caption("Estimasi ini hanya ilustrasi, bukan prediksi yang dapat diandalkan.")


# ==========================================================
# PEMUATAN DATA (dipakai semua tab)
# ==========================================================
def widget_unggah():
    """Unggah CSV di tab Data; isinya disimpan di session_state agar tab lain ikut memakainya."""
    with st.expander("Ganti sumber data (opsional)"):
        unggah = st.file_uploader("Unggah CSV hasil notebook. Jika kosong, memakai datas/dataset_hasil.csv",
                                  type=['csv'])
        if unggah is not None:
            st.session_state['unggahan'] = (unggah.name, unggah.getvalue())
        if 'unggahan' in st.session_state and st.button("Kembali ke dataset bawaan"):
            del st.session_state['unggahan']
            st.rerun()


def muat_data():
    """Mengembalikan (df, nama_berkas, tabel_cek_kolom, tweet_url_dibuang). df = None bila data tidak valid."""
    if 'unggahan' in st.session_state:
        nama, isi = st.session_state['unggahan']
    elif PATH_DATA.exists():
        isi, nama = PATH_DATA.read_bytes(), PATH_DATA.name
    else:
        st.error(f"Berkas {PATH_DATA.name} tidak ditemukan di folder datas/. Letakkan berkas itu atau unggah CSV.")
        return None, None, None, False

    try:
        mentah = baca_csv(isi)
    except Exception as e:
        st.error(f"CSV tidak dapat dibaca: {e}")
        return None, nama, None, False
    if mentah.empty:
        st.error("CSV tidak berisi baris data.")
        return None, nama, None, False

    # pemeriksaan kolom wajib
    cek = pd.DataFrame({'Kolom wajib': KOLOM_WAJIB,
                        'Status': ['ada' if k in mentah.columns else 'tidak ada' for k in KOLOM_WAJIB]})
    kurang = [k for k in KOLOM_WAJIB if k not in mentah.columns]
    if kurang:
        st.error(f"Kolom wajib tidak ditemukan: {kurang}. Aplikasi tidak dapat memproses data ini.")
        st.dataframe(cek, hide_index=True)
        return None, nama, cek, False

    df = mentah.copy()
    url_dibuang = 'tweet_url' in df.columns
    if url_dibuang:
        df = df.drop(columns=['tweet_url'])
    return df, nama, cek, url_dibuang


# ==========================================================
# TAB 1: DATA
# ==========================================================
def render_tab_data(df, nama, cek, url_dibuang):
    if url_dibuang:
        st.warning("Kolom 'tweet_url' memuat nama akun, jadi disembunyikan. Sebaiknya dibuang dari berkas sebelum dikumpulkan.")
    waktu = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    st.write("### Ringkasan Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah postingan", f"{len(df):,}")
    c2.metric("Jumlah kolom", df.shape[1])
    if waktu.notna().any():
        c3.metric("Rentang tanggal", f"{waktu.min():%Y-%m-%d} s.d. {waktu.max():%Y-%m-%d}")
    st.caption(f"Sumber data: {nama}")

    with st.expander("Pemeriksaan kolom"):
        st.dataframe(cek, hide_index=True)
        opsional = {'text_clean': 'clustering & word cloud', 'sentimen': 'tab Sentimen', 'cluster': 'tab Clustering',
                    'query': 'sebaran per kata kunci'}
        hilang = {k: v for k, v in opsional.items() if k not in df.columns}
        if hilang:
            st.info("Kolom opsional yang tidak ada (fitur terkait dibatasi): "
                    + "; ".join(f"`{k}` → {v}" for k, v in hilang.items()))
        else:
            st.write("Semua kolom opsional tersedia.")

    # filter
    st.write("### Filter Data")
    f1, f2, f3 = st.columns([2, 1, 1])
    with f1:
        kata_cari = st.text_input("Cari kata kunci teks:", placeholder="Ketik kata kunci...")
    with f2:
        if 'sentimen' in df.columns:
            pilihan = ["Semua"] + sorted(str(s) for s in df['sentimen'].dropna().unique())
            f_sent = st.selectbox("Filter Sentimen:", pilihan)
        else:
            f_sent = "Semua"
    with f3:
        if 'cluster' in df.columns:
            pilihan = ["Semua"] + sorted(str(c) for c in df['cluster'].dropna().unique())
            f_clu = st.selectbox("Filter Cluster:", pilihan)
        else:
            f_clu = "Semua"

    tampil = df
    if kata_cari:
        tampil = tampil[tampil['text'].astype(str).str.contains(kata_cari, case=False, na=False, regex=False)]
    if f_sent != "Semua":
        tampil = tampil[tampil['sentimen'].astype(str) == f_sent]
    if f_clu != "Semua":
        tampil = tampil[tampil['cluster'].astype(str) == f_clu]
    st.caption(f"Menampilkan **{len(tampil)}** dari **{len(df)}** postingan.")
    kolom_tampil = [c for c in ['created_at', 'query', 'sentimen', 'cluster', 'text', 'like_count'] if c in tampil.columns]
    st.dataframe(tampil[kolom_tampil], width='stretch', height=320)

    # baris yang dikeluarkan
    with st.expander("Baris yang dikeluarkan dari analisis tertentu"):
        n = len(df)
        d_s, k_s = siapkan_sentimen(df)
        d_c, k_c = siapkan_cluster(df)
        rr = baris_regresi(df)
        baris = [
            {'Analisis': 'Sentimen', 'Dipakai': 0 if d_s is None else n - k_s, 'Dikeluarkan': n if d_s is None else k_s,
             'Alasan': "kolom 'sentimen' tidak ada" if d_s is None else "label sentimen kosong/tidak valid"},
            {'Analisis': 'Clustering', 'Dipakai': 0 if d_c is None else n - k_c, 'Dikeluarkan': n if d_c is None else k_c,
             'Alasan': "kolom 'cluster' tidak ada" if d_c is None else "label cluster kosong"},
            {'Analisis': 'Regresi', 'Dipakai': rr['dipakai'],
             'Dikeluarkan': rr['like_tidak_valid'] + rr['teks_kosong'],
             'Alasan': f"likes kosong/bukan angka/negatif ({rr['like_tidak_valid']}) dan teks kosong "
                       f"({rr['teks_kosong']}); likes tidak diganti 0"},
        ]
        st.dataframe(pd.DataFrame(baris), hide_index=True, width='stretch')

    with st.expander("Informasi pengambilan dan pembersihan data"):
        st.dataframe(pd.DataFrame(list(INFO_PENGAMBILAN.items()), columns=['Item', 'Keterangan']),
                     hide_index=True, width='stretch')
        tahap = baca_opsional(PATH_TAHAP)
        if tahap is None:
            st.info("Berkas jumlah_data_per_tahap.csv tidak ditemukan (opsional).")
        else:
            st.write("**Jumlah data pada setiap tahap pembersihan:**")
            st.dataframe(tahap, hide_index=True, width='stretch')


# ==========================================================
# TAB 2: SENTIMEN
# ==========================================================
def render_tab_sentimen(df):
    st.subheader("Analisis Sentimen (IndoBERT)")
    d, n_keluar = siapkan_sentimen(df)
    if d is None:
        st.error("Kolom 'sentimen' tidak ada pada data. Gunakan dataset_hasil.csv dari notebook; "
                 "klasifikasi dijalankan di notebook agar aplikasi tetap ringan.")
        return
    if d.empty:
        st.error("Tidak ada baris dengan label sentimen yang valid (positif/netral/negatif).")
        return
    if n_keluar:
        st.warning(f"{n_keluar} baris dikeluarkan karena label sentimennya kosong atau tidak valid.")
    st.caption("Model: mdhugol/indonesia-bert-sentiment-classification. Klasifikasi dijalankan di notebook "
               "pada teks asli postingan; aplikasi membaca hasilnya dari CSV.")

    hitung = d['sentimen'].value_counts().reindex(URUTAN_SENTIMEN, fill_value=0)
    persen = (hitung / hitung.sum() * 100).round(1)

    st.write("### Jumlah dan Distribusi Sentimen")
    kiri, kanan = st.columns([1, 3])
    with kiri:
        st.dataframe(pd.DataFrame({'Jumlah': hitung, 'Persentase (%)': persen}), width='stretch')
    with kanan:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        warna = [WARNA_SENTIMEN[s] for s in hitung.index]
        ax[0].bar(hitung.index, hitung.values, width=0.6, color=warna)
        for i, s in enumerate(hitung.index):
            ax[0].text(i, hitung[s], f"{hitung[s]} ({persen[s]}%)", ha='center', va='bottom', fontsize=10)
        ax[0].set_title('Distribusi Sentimen Postingan')
        ax[0].set_ylabel('Jumlah postingan')
        ax[0].set_ylim(0, max(hitung.max(), 1) * 1.18)
        ax[0].grid(axis='x', visible=False)
        ax[1].pie(hitung, labels=hitung.index, autopct='%1.1f%%', colors=warna, startangle=90)
        ax[1].set_title('Persentase sentimen')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.write("### Tabel Postingan dan Hasil Klasifikasi")
    pilih = st.selectbox("Tampilkan sentimen:", ["Semua"] + URUTAN_SENTIMEN, key='filter_tabel_sentimen')
    tabel = d if pilih == "Semua" else d[d['sentimen'] == pilih]
    kolom = [c for c in ['text', 'sentimen', 'skor_sentimen'] if c in tabel.columns]
    st.caption(f"{len(tabel)} postingan")
    st.dataframe(tabel[kolom], width='stretch', height=320)

    # word cloud
    st.write("### Word Cloud untuk Setiap Sentimen")
    if 'text_clean' not in d.columns:
        st.info("Kolom 'text_clean' tidak tersedia, word cloud tidak dapat dibuat.")
    else:
        sw, _ = muat_stopwords()
        sw_wc = tuple(sorted(set(sw) | set(KATA_KUNCI_WORDCLOUD)))
        awan = buat_wordcloud(tuple(d['text_clean'].fillna('').astype(str)), tuple(d['sentimen']), sw_wc)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        for i, s in enumerate(['negatif', 'netral', 'positif']):
            if awan[s] is not None:
                ax[i].imshow(awan[s], interpolation='bilinear')
            else:
                ax[i].text(0.5, 0.5, 'Tidak ada tweet', ha='center', va='center')
            ax[i].set_title(f'Word cloud sentimen {s}')
            ax[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Kata kunci pengumpulan data (hak, cipta, desainer, ilustrator, seniman, penulis, musisi) "
                   "dihilangkan dari word cloud agar kata lain terlihat. Ini hanya pilihan tampilan.")

    # pemeriksaan manual
    st.write("### Ringkasan Pemeriksaan Manual")
    ev = baca_opsional(PATH_EVAL)
    if ev is None:
        st.warning("Berkas evaluasi_sentimen.csv tidak ditemukan.")
        return
    ev.columns = ev.columns.str.strip().str.lower()
    if not {'label_manual', 'prediksi_model'}.issubset(ev.columns):
        st.error("Berkas evaluasi harus memuat kolom 'label_manual' dan 'prediksi_model'.")
        return
    ev['label_manual'] = ev['label_manual'].astype(str).str.strip().str.lower()
    ev['prediksi_model'] = ev['prediksi_model'].astype(str).str.strip().str.lower()
    ev['sesuai'] = ev['label_manual'] == ev['prediksi_model']
    n, k = len(ev), int(ev['sesuai'].sum())
    lo, hi = wilson(k, n)
    m1, m2, m3 = st.columns(3)
    m1.metric("Sesuai", f"{k} dari {n}")
    m2.metric("Persentase", f"{k / n * 100:.0f}%")
    m3.metric("Interval 95%", f"{lo * 100:.0f}% - {hi * 100:.0f}%")
    dasar = ev['label_manual'].value_counts()
    st.caption(f"Pembanding: label manual terbanyak adalah '{dasar.index[0]}' ({dasar.iloc[0]} dari {n} sampel, "
               f"{dasar.iloc[0] / n * 100:.0f}%), jadi menebak kelas itu saja sudah mencapai {dasar.iloc[0] / n * 100:.0f}%.")
    st.caption("Ini evaluasi terbatas pada sampel acak postingan. Hasilnya belum menggambarkan akurasi model "
               "pada seluruh percakapan di X.")
    silang = pd.crosstab(ev['label_manual'], ev['prediksi_model']).reindex(
        index=URUTAN_SENTIMEN, columns=URUTAN_SENTIMEN, fill_value=0)
    silang.index.name, silang.columns.name = 'Label manual (baris)', 'Prediksi model (kolom)'
    st.dataframe(silang, width='stretch')
    beda = ev[~ev['sesuai']]
    st.write(f"**Ketidaksesuaian ({len(beda)} postingan):**")
    kolom_beda = [c for c in ['no', 'text', 'label_manual', 'prediksi_model', 'skor_sentimen'] if c in beda.columns]
    st.dataframe(beda[kolom_beda], hide_index=True, width='stretch')


# ==========================================================
# TAB 3: CLUSTERING
# ==========================================================
def render_tab_cluster(df):
    st.subheader("Clustering Teks (TF-IDF + K-Means, K = 3)")
    d, n_keluar = siapkan_cluster(df)
    if d is None or d.empty:
        st.error("Kolom 'cluster' tidak ada atau kosong. Gunakan dataset_hasil.csv dari notebook.")
        return
    if 'text_clean' not in d.columns:
        st.error("Kolom 'text_clean' diperlukan untuk menampilkan kata dominan dan peta cluster.")
        return
    if n_keluar:
        st.warning(f"{n_keluar} baris dikeluarkan karena label cluster kosong.")

    sw, nltk_ok = muat_stopwords()
    if not nltk_ok:
        st.warning("Stopwords tidak tersedia (stopwords.txt tidak ada dan NLTK tidak dapat diunduh). Kata dominan dan "
                   "peta bisa sedikit berbeda dari notebook. Letakkan stopwords.txt di folder datas/.")
    stop_cluster = tuple(sorted(w for w in sw if re.fullmatch(r'[a-z]{3,}', w)))
    d = d.reset_index(drop=True)
    koordinat, info, sil, tanpa_kata = analisis_cluster(
        tuple(d['text_clean'].fillna('').astype(str)), tuple(d['cluster']), stop_cluster)

    # ringkasan per cluster
    ring = baca_opsional(PATH_RINGKASAN)
    kata_csv = {}
    if ring is not None and {'cluster', 'lima_kata_dominan'}.issubset(ring.columns):
        kata_csv = {int(r['cluster']): str(r['lima_kata_dominan']) for _, r in ring.iterrows()}

    jumlah = d['cluster'].value_counts().sort_index()
    baris = []
    for c in sorted(info):
        baris.append({'Cluster': c, 'Tema': d.loc[d['cluster'] == c, 'tema'].iloc[0],
                      'Jumlah postingan': int(jumlah[c]), 'Persentase (%)': round(jumlah[c] / len(d) * 100, 1),
                      'Lima kata dominan': kata_csv.get(c, ', '.join(info[c]['top5']))})
    st.write("### Ringkasan Cluster")
    st.dataframe(pd.DataFrame(baris), hide_index=True, width='stretch')
    st.caption(f"Nama tema ditentukan setelah membaca isi postingan, bukan sebelum clustering. "
               f"Postingan tanpa kata tersisa setelah stopwords dibuang: {tanpa_kata} (tetap dihitung sebagai anggota cluster).")

    st.write("### Tiga Contoh Postingan per Cluster (paling dekat ke pusat cluster)")
    for c in sorted(info):
        tema = d.loc[d['cluster'] == c, 'tema'].iloc[0]
        with st.expander(f"Cluster {c}: {tema}  ({int(jumlah[c])} postingan)"):
            st.write("**Kata dominan:** " + kata_csv.get(c, ', '.join(info[c]['top5'])))
            for idx in info[c]['contoh']:
                st.write("- " + str(d.loc[idx, 'text'])[:300].replace('\n', ' '))

    # peta cluster
    st.write("### Peta Postingan per Cluster")
    st.caption("TF-IDF diringkas ke 2 dimensi dengan TruncatedSVD (sama dengan notebook).")
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in sorted(info):
        m = (d['cluster'] == c).values
        ax.scatter(koordinat[m, 0], koordinat[m, 1], s=18, alpha=0.7, color=WARNA_CLUSTER[c % len(WARNA_CLUSTER)],
                   edgecolors='white', linewidths=0.5, label=f"{c}: {d.loc[m, 'tema'].iloc[0]}")
    ax.set_xlabel('Komponen 1')
    ax.set_ylabel('Komponen 2')
    ax.legend(title='Cluster', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # kualitas dan tumpang tindih
    c1, c2 = st.columns(2)
    with c1:
        st.write("### Ukuran Cluster")
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.bar([str(c) for c in jumlah.index], jumlah.values,
               color=[WARNA_CLUSTER[c % len(WARNA_CLUSTER)] for c in jumlah.index])
        for i, v in enumerate(jumlah.values):
            ax.text(i, v, str(int(v)), ha='center', va='bottom')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Jumlah postingan')
        ax.grid(axis='x', visible=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.write("### Kualitas dan Tumpang Tindih")
        st.metric("Silhouette score (cosine)", f"{sil:.3f}")
        st.caption("Nilai mendekati 0 berarti cluster saling tumpang tindih dan belum terpisah tegas.")
        ids = sorted(info)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                bersama = sorted(set(info[a]['top10']) & set(info[b]['top10']))
                st.write(f"Kata dominan (10 teratas) bersama cluster {a} & {b}: " + (', '.join(bersama) if bersama else '-'))

    if 'sentimen' in d.columns:
        ds, _ = siapkan_sentimen(d)
        st.write("### Sebaran Sentimen per Tema (% per baris)")
        silang = pd.crosstab(ds['tema'], ds['sentimen'], normalize='index') * 100
        st.dataframe(silang.reindex(columns=URUTAN_SENTIMEN, fill_value=0).round(1), width='stretch')

    if 'query' in d.columns:
        st.write("### Sebaran Cluster per Kata Kunci Crawling (% per cluster)")
        ct = (pd.crosstab(d['cluster'], d['query'], normalize='index') * 100).round(0)
        st.dataframe(ct, width='stretch')
        rata = kesesuaian_query(d)
        if rata is not None:
            st.caption(f"Rata-rata {rata:.0f}% postingan dari tiap kata kunci jatuh ke satu cluster yang sama. "
                       "Semakin tinggi angkanya, semakin besar kemungkinan cluster mencerminkan cara data dikumpulkan.")


# ==========================================================
# TAB 4: REGRESI
# ==========================================================
def render_tab_regresi(df):
    st.subheader("Regresi Linear: Memperkirakan Jumlah Likes")
    reg = muat_regresi()
    for c in reg['catatan']:
        st.warning(c)
    p = reg['p']
    if p is None:
        st.error("Hasil regresi tidak ditemukan. Jalankan sel ekspor di notebook, lalu letakkan model_regresi.joblib "
                 "(atau parameter_model.json dan prediksi_data_uji.csv) di folder datas/.")
        return

    meta, sel, u, dasar = p['meta'], p['seleksi_fitur'], p['model_utama'], p['pembanding_dasar']
    model_u = reg['objek'].get('utama')
    model_k = reg['objek'].get('ketentuan')

    st.caption(f"Model dilatih di notebook pada dataset yang sama: data latih {meta['n_latih']} baris (80%), data uji "
               f"{meta['n_uji']} baris (20%), random_state = {meta['random_state']}. Semua angka evaluasi dari data uji. "
               "Hasil regresi tidak dihitung ulang untuk CSV yang diunggah.")
    n_model = meta['n_latih'] + meta['n_uji']
    if len(df) != n_model:
        st.warning(f"Jumlah baris data di aplikasi ({len(df)}) berbeda dari data pelatihan model ({n_model}). "
                   "Hasil regresi di bawah berasal dari notebook dan mungkin tidak sesuai dengan data ini.")

    # 1) pemilihan fitur
    st.write("### Pemilihan Fitur (Korelasi Linear)")
    st.caption("Korelasi Pearson dihitung hanya dari data latih, sebelum fitur dipilih, supaya data uji tidak ikut "
               "memengaruhi pilihan. Dua variabel dengan |r| tertinggi dipilih sebagai fitur.")
    kor = sel['korelasi_dengan_likes']
    st.dataframe(pd.DataFrame({
        'Variabel': [NAMA_FITUR.get(k, k) for k in kor],
        'Pearson (r) dengan likes': [round(v, 3) for v in kor.values()],
        'Dipilih': ['ya' if k in u['fitur'] else '' for k in kor],
    }), hide_index=True, width='stretch')
    fig = gambar_heatmap_param(sel)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Pegangan membaca |r| (konvensi umum): di bawah 0,1 diabaikan; 0,1–0,3 lemah; 0,3–0,5 sedang; "
               "di atas 0,5 kuat. Baca baris atau kolom 'Jumlah likes' untuk hubungan dengan likes; angka lain adalah "
               "korelasi antar-variabel. Korelasi tidak menunjukkan sebab-akibat.")
    st.success("Fitur terpilih: " + ", ".join(u['label_fitur']))

    # 2) evaluasi
    st.write("### Model dan Evaluasi")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{u['mae']:.2f} likes")
    m2.metric("R²", f"{u['r2']:.3f}")
    m3.metric("MAE tebak rata-rata", f"{dasar['mae_tebak_rata']:.2f}")
    m4.metric("MAE tebak median", f"{dasar['mae_tebak_median']:.2f}")
    st.code(f"likes ≈ {u['intercept']:.2f}" + "".join(
        f" + ({c:.3f} × {f})" for c, f in zip(u['koefisien'], u['fitur'])), language=None)
    st.caption(f"Prediksi negatif pada data uji: {u['n_negatif']} dari {meta['n_uji']}.")

    # 3) grafik dan tabel data uji
    uji = reg['uji']
    if uji is None or 'prediksi_utama' not in uji.columns:
        st.info("Data prediksi uji tidak tersedia (prediksi_data_uji.csv), sehingga grafik aktual vs prediksi tidak dapat ditampilkan.")
    else:
        uji = uji.copy()
        st.write("### Likes Aktual vs Prediksi (Data Uji)")
        kiri, kanan = st.columns([1.1, 1])
        with kiri:
            log_skala = st.checkbox("Skala logaritmik (symlog), agar titik tidak menumpuk", value=True)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            ax.scatter(uji['like_aktual'], uji['prediksi_utama'], s=22, alpha=0.7, color='#2a78d6',
                       edgecolors='white', linewidths=0.5, label='Postingan (data uji)')
            batas = [min(uji['like_aktual'].min(), uji['prediksi_utama'].min()),
                     max(uji['like_aktual'].max(), uji['prediksi_utama'].max())]
            ax.plot(batas, batas, '--', color='#52514e', linewidth=1.5, label='Prediksi sempurna (aktual = prediksi)')
            ax.axhline(0, color='#e34948', linewidth=1, alpha=0.6, label='Likes = 0 (di bawahnya = prediksi negatif)')
            if log_skala:
                ax.set_xscale('symlog')
                ax.set_yscale('symlog')
            ax.set_xlabel('Likes aktual')
            ax.set_ylabel('Likes prediksi')
            ax.set_title('Likes aktual vs prediksi (data uji)')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        with kanan:
            st.write("**Tabel data uji:**")
            st.dataframe(uji.drop(columns=['text'], errors='ignore').round(2), width='stretch', height=380)

        with st.expander("Lima postingan dengan selisih prediksi terbesar (data uji)"):
            t = uji.copy()
            t['selisih'] = (t['like_aktual'] - t['prediksi_utama']).abs().round(1)
            st.dataframe(t.sort_values('selisih', ascending=False).head(5), hide_index=True, width='stretch')
            st.caption("Selisih terbesar umumnya berasal dari postingan dengan likes sangat tinggi.")

    # 4) perbandingan model
    pembanding = [{'Model': 'Terpilih: ' + ', '.join(u['label_fitur']), 'MAE': u['mae'], 'R²': u['r2'],
                   'Prediksi negatif': u['n_negatif']}]
    if 'model_ketentuan' in p:
        k = p['model_ketentuan']
        pembanding.append({'Model': 'Fitur ketentuan tugas: ' + ', '.join(k['label_fitur']), 'MAE': k['mae'],
                           'R²': k['r2'], 'Prediksi negatif': k['n_negatif']})
    pembanding += [{'Model': 'Tebak rata-rata', 'MAE': dasar['mae_tebak_rata'], 'R²': None, 'Prediksi negatif': None},
                   {'Model': 'Tebak median', 'MAE': dasar['mae_tebak_median'], 'R²': None, 'Prediksi negatif': None}]
    st.write("### Perbandingan Model")
    st.dataframe(pd.DataFrame(pembanding).round(3), hide_index=True, width='stretch')

    # 5) interpretasi
    st.write("### Interpretasi")
    st.markdown(
        f"- **MAE {u['mae']:.2f}**: rata-rata, perkiraan model meleset sekitar {u['mae']:.0f} likes dari jumlah likes "
        f"sebenarnya (satuan sama dengan likes). Median likes data uji {meta['median_likes_uji']:.0f}.\n"
        f"- **R² {u['r2']:.3f}**: 1 berarti sempurna, 0 berarti sama baiknya dengan menebak rata-rata, dan **negatif "
        f"berarti lebih buruk daripada menebak rata-rata**.\n"
        f"- **Keterbatasan**: likes juga dipengaruhi jangkauan akun dan isi pesan yang tidak ada dalam model, dan "
        f"sebaran likes sangat miring (median seluruh data {meta['median_likes_semua']:.0f})."
    )
    if u['r2'] < 0:
        st.warning("R² negatif: pada data uji, model ini tidak lebih baik daripada menebak rata-rata likes.")
    elif dasar['mae_tebak_median'] < u['mae']:
        st.info(f"Menebak median likes (MAE {dasar['mae_tebak_median']:.2f}) lebih akurat daripada model ({u['mae']:.2f}).")
    if u['metrik_respons']:
        st.warning("Fitur terpilih adalah metrik respons yang terukur bersamaan dengan likes. Model ini menggambarkan "
                   "hubungan antar-metrik engagement, bukan prediksi likes sebelum postingan terbit.")
    st.caption("Regresi linear dapat menghasilkan estimasi negatif; nilai itu tidak bermakna sebagai jumlah likes dan "
               "menunjukkan keterbatasan model.")

    # 6) formulir estimasi
    st.write("### Estimasi Likes")
    form_estimasi(u, model_u, 'utama')

    # 7) model sesuai ketentuan tugas (pembanding)
    if 'model_ketentuan' in p:
        k = p['model_ketentuan']
        with st.expander("Model dengan fitur ketentuan tugas (jumlah kata + hashtag), sebagai pembanding"):
            k1, k2 = st.columns(2)
            k1.metric("MAE", f"{k['mae']:.2f} likes")
            k2.metric("R²", f"{k['r2']:.3f}")
            st.code(f"likes ≈ {k['intercept']:.2f}" + "".join(
                f" + ({c:.3f} × {f})" for c, f in zip(k['koefisien'], k['fitur'])), language=None)
            st.caption("Dilatih dengan pembagian data yang sama. Prediksi negatif: "
                       f"{k['n_negatif']} dari {meta['n_uji']}.")
            form_estimasi(k, model_k, 'ketentuan')


# ==========================================================
# TEMUAN DAN KETERBATASAN
# ==========================================================
def render_temuan(df):
    st.subheader("Temuan Utama dan Keterbatasan")
    temuan = []

    d, _ = siapkan_sentimen(df)
    if d is not None and len(d):
        p_s = (d['sentimen'].value_counts(normalize=True) * 100).reindex(URUTAN_SENTIMEN, fill_value=0)
        urut = p_s.sort_values(ascending=False)
        kal = (f"**Sentimen.** Dari {len(d)} postingan, kategori terbanyak adalah *{urut.index[0]}* "
               f"({urut.iloc[0]:.1f}%). Sebaran: positif {p_s['positif']:.1f}%, netral {p_s['netral']:.1f}%, "
               f"negatif {p_s['negatif']:.1f}%.")
        ev = ringkas_evaluasi()
        if ev:
            kal += (f" Pada pemeriksaan manual {ev['n']} sampel, {ev['k']} sesuai ({ev['k'] / ev['n'] * 100:.0f}%); "
                    f"menebak label manual terbanyak ('{ev['kelas']}') saja sudah mencapai {ev['dasar']:.0f}%, "
                    "sehingga kesesuaian ini perlu dibaca hati-hati.")
        temuan.append(kal)

    dc, _ = siapkan_cluster(df)
    if dc is not None and len(dc) and 'text_clean' in dc.columns:
        sw, _ = muat_stopwords()
        stop_cluster = tuple(sorted(w for w in sw if re.fullmatch(r'[a-z]{3,}', w)))
        dc = dc.reset_index(drop=True)
        _, _, sil, _ = analisis_cluster(tuple(dc['text_clean'].fillna('').astype(str)), tuple(dc['cluster']), stop_cluster)
        rata = kesesuaian_query(dc)
        kalimat = f"**Clustering.** Silhouette score {sil:.3f}"
        if rata is not None and rata >= 70:
            kalimat += (f", dan rata-rata {rata:.0f}% postingan dari tiap kata kunci crawling jatuh ke satu cluster yang sama. "
                        "Cluster sebagian besar mencerminkan cara data dikumpulkan dan belum menunjukkan tiga tema yang tegas.")
        elif rata is not None:
            kalimat += f", dan {rata:.0f}% postingan dari tiap kata kunci rata-rata berada di satu cluster."
        else:
            kalimat += "; cluster saling tumpang tindih bila nilainya mendekati 0."
        temuan.append(kalimat)

    reg = muat_regresi()
    p = reg['p']
    if p:
        u, dasar = p['model_utama'], p['pembanding_dasar']
        n_kand = len(p['seleksi_fitur']['korelasi_dengan_likes'])
        kal_reg = (f"**Regresi.** Dari {n_kand} variabel kandidat, dua dengan korelasi Pearson tertinggi terhadap likes "
                   f"({', '.join(u['label_fitur'])}) dipilih sebagai fitur. Pada data uji, modelnya mencapai MAE "
                   f"{u['mae']:.1f} likes dan R² {u['r2']:.3f}; menebak median likes menghasilkan MAE "
                   f"{dasar['mae_tebak_median']:.1f}.")
        if u['metrik_respons']:
            kal_reg += (" Fitur ini terukur bersamaan dengan likes, sehingga model menggambarkan hubungan antar-metrik "
                        "engagement, bukan prediksi sebelum posting.")
        if 'model_ketentuan' in p:
            kal_reg += (f" Sebagai pembanding, model jumlah kata dan hashtag hanya mencapai R² "
                        f"{p['model_ketentuan']['r2']:.3f}.")
        temuan.append(kal_reg)

    if temuan:
        st.write("### 3 Temuan Utama")
        for i, t in enumerate(temuan, 1):
            st.markdown(f"{i}. {t}")
    else:
        st.info("Temuan belum dapat dihitung karena kolom analisis belum tersedia pada data.")

    waktu = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    periode = f"{waktu.min():%Y-%m-%d} s.d. {waktu.max():%Y-%m-%d}" if waktu.notna().any() else "periode tidak diketahui"
    if p and p['model_utama']['metrik_respons']:
        k2 = (f"fitur terpilih ({', '.join(p['model_utama']['label_fitur'])}) terukur bersamaan dengan likes, sehingga "
              "model menggambarkan hubungan antar-metrik engagement, bukan prediksi sebelum posting. Jangkauan akun dan "
              "isi pesan tidak ada dalam model, dan sebaran likes sangat miring.")
    else:
        k2 = "likes dipengaruhi usia postingan, jangkauan akun, dan isi pesan yang tidak ada dalam model, dan sebaran likes sangat miring."
    r_kata = p['seleksi_fitur']['korelasi_dengan_likes'].get('jumlah_kata') if p else None
    if r_kata is not None:
        k2 += f" Ciri teks sederhana seperti jumlah kata hampir tidak berkorelasi dengan likes (r = {r_kata:.2f})."
    st.write("### 2 Keterbatasan")
    st.markdown(
        f"1. **Data sampel.** Data berasal dari pencarian kata kunci tertentu ({periode}), bukan seluruh percakapan di X, "
        "sehingga tidak mewakili opini masyarakat. Sentimen diberi label oleh model pra-latih dan hanya diperiksa manual pada "
        "sejumlah kecil sampel; sebagian postingan tidak relevan atau berbahasa lain masih mungkin ada.\n"
        f"2. **Model likes sederhana.** Pada model regresi, {k2}"
    )


# ==========================================================
# TATA LETAK UTAMA
# ==========================================================
def main():
    st.title("Dashboard Analisis AI pada Profesi Kreatif")
    st.caption("Analisis sentimen, clustering, dan regresi linear pada postingan X")

    menu = ["Data", "Sentimen", "Clustering", "Regresi"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Data":
        widget_unggah()
        data, nama_berkas, tabel_cek, url_dibuang = muat_data()
        if data is not None:
            render_tab_data(data, nama_berkas, tabel_cek, url_dibuang)
    else:
        data, nama_berkas, tabel_cek, url_dibuang = muat_data()
        if data is None:
            st.info("Muat data yang valid di menu Data terlebih dahulu.")
            return

        if choice == "Sentimen":
            render_tab_sentimen(data)
        elif choice == "Clustering":
            render_tab_cluster(data)
        elif choice == "Regresi":
            render_tab_regresi(data)

    if data is not None:
        st.divider()
        render_temuan(data)


if __name__ == "__main__":
    main()