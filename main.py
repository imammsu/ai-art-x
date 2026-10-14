import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, pairwise_distances, r2_score, silhouette_score
from sklearn.model_selection import train_test_split

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

INFO_PENGAMBILAN = {
    'Sumber': 'Platform X, dikumpulkan dengan tweet-harvest (Google Colab)',
    'Kata kunci': 'ilustrator ai, hak cipta ai, seniman ai, penulis ai, desainer ai, musisi ai',
    'Penyaringan periode': '2025-01-01 s.d. 2026-10-11 (kolom created_at)',
    'Waktu pengambilan': 'ISI sesuai catatan crawling kelompok (sertakan zona waktu)',   # ✏️
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
FITUR = ['jumlah_kata', 'jumlah_hashtag']
FITUR_ENG = ['retweet_count', 'reply_count']
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


@st.cache_data(show_spinner='Melatih model regresi...')
def latih_regresi(data: pd.DataFrame):
    """Prosedur sama dengan notebook: fitur dari teks asli, 80/20, random_state=42, latih hanya di data latih."""
    d = data.copy()
    d['text'] = d['text'].fillna('').astype(str)
    d['like_count'] = pd.to_numeric(d['like_count'], errors='coerce')
    n_awal = len(d)

    tidak_valid = d['like_count'].isna() | (d['like_count'] < 0)
    n_like_tidak_valid = int(tidak_valid.sum())
    d = d[~tidak_valid]
    kosong = d['text'].str.strip() == ''
    n_teks_kosong = int(kosong.sum())
    d = d[~kosong].reset_index(drop=True)          # likes tidak diketahui TIDAK diganti 0
    if len(d) < 20:
        return {'ok': False, 'pesan': f'Data valid hanya {len(d)} baris (minimal 20 untuk membagi data latih/uji).',
                'n_awal': n_awal, 'n_like_tidak_valid': n_like_tidak_valid, 'n_teks_kosong': n_teks_kosong}

    d['jumlah_kata'] = d['text'].str.split().str.len()
    d['jumlah_hashtag'] = d['text'].str.count(r'#\w+')
    X, y = d[FITUR], d['like_count']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r = {
        'ok': True, 'n_awal': n_awal, 'n_like_tidak_valid': n_like_tidak_valid, 'n_teks_kosong': n_teks_kosong,
        'n_pakai': len(d), 'n_latih': len(X_tr), 'n_uji': len(X_te),
        'model': model, 'intercept': float(model.intercept_), 'coef': [float(c) for c in model.coef_],
        'mae': float(mean_absolute_error(y_te, y_pred)), 'r2': float(r2_score(y_te, y_pred)),
        'mae_rata': float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.mean()))),
        'mae_median': float(mean_absolute_error(y_te, np.full(len(y_te), y_tr.median()))),
        'median_uji': float(y_te.median()), 'median_semua': float(y.median()),
        'n_negatif': int((y_pred < 0).sum()),
        'y_test': y_te.to_numpy(), 'y_pred': y_pred, 'X_test': X_te.reset_index(drop=True),
        'kata_median': int(d['jumlah_kata'].median()), 'kata_min': int(d['jumlah_kata'].min()),
        'kata_max': int(d['jumlah_kata'].max()), 'hashtag_max': int(d['jumlah_hashtag'].max()),
        'persen_berhashtag': float((d['jumlah_hashtag'] > 0).mean() * 100),
        'data_latih': pd.concat([X_tr, y_tr], axis=1),
        'eks': None,
        'teks_uji': d.loc[X_te.index, 'text'].tolist(),
        'deskripsi': d[FITUR + ['like_count']].describe().round(2),
        'sebaran_hashtag': d['jumlah_hashtag'].value_counts().sort_index(),
        'n_hashtag_latih': int((X_tr['jumlah_hashtag'] > 0).sum()),
    }
    r['est_tipikal'] = float(model.predict(pd.DataFrame([[r['kata_median'], 0]], columns=FITUR))[0])

    # model tambahan (cell 63 notebook): retweet + reply -> likes, skala asli,
    # baris latih/uji SAMA dengan model utama
    if all(k in d.columns for k in FITUR_ENG):
        d[FITUR_ENG] = d[FITUR_ENG].apply(pd.to_numeric, errors='coerce')
    if all(k in d.columns for k in FITUR_ENG) and d[FITUR_ENG].notna().all().all():
        X2_tr, X2_te = d.loc[X_tr.index, FITUR_ENG], d.loc[X_te.index, FITUR_ENG]
        m2 = LinearRegression().fit(X2_tr, y_tr)
        p2 = m2.predict(X2_te)
        r['eks'] = {'model': m2, 'mae': float(mean_absolute_error(y_te, p2)), 'r2': float(r2_score(y_te, p2)),
                    'intercept': float(m2.intercept_), 'coef': [float(c) for c in m2.coef_],
                    'n_negatif': int((p2 < 0).sum()), 'y_pred': p2,
                    'X_test': X2_te.reset_index(drop=True),
                    'data_latih': pd.concat([X2_tr, y_tr], axis=1),
                    'deskripsi': d[FITUR_ENG + ['like_count']].describe().round(2),
                    'maks': {k: int(d[k].max()) for k in FITUR_ENG},
                    'median': {k: float(d[k].median()) for k in FITUR_ENG}}
    return r


def hasil_regresi(df):
    kolom = ['text', 'like_count'] + [k for k in FITUR_ENG if k in df.columns]
    return latih_regresi(df[kolom])


# ==========================================================
# PEMUATAN DATA (dipakai semua halaman)
# ==========================================================
def widget_unggah():
    """Unggah CSV di halaman Dataset; isinya disimpan di session_state agar halaman lain ikut memakainya."""
    with st.expander("Ganti sumber data (opsional)"):
        unggah = st.file_uploader("Unggah CSV hasil notebook. Jika kosong, memakai datas/dataset_hasil.csv",
                                  type=['csv'])
        if unggah is not None:
            st.session_state['unggahan'] = (unggah.name, unggah.getvalue())
        if 'unggahan' in st.session_state and st.button("Kembali ke dataset bawaan"):
            del st.session_state['unggahan']
            st.rerun()


def muat_data():
    """Mengembalikan (df, nama_berkas, tabel_cek_kolom). df = None bila data tidak valid."""
    if 'unggahan' in st.session_state:
        nama, isi = st.session_state['unggahan']
    elif PATH_DATA.exists():
        isi, nama = PATH_DATA.read_bytes(), PATH_DATA.name
    else:
        st.error(f"Berkas {PATH_DATA.name} tidak ditemukan di folder datas/. Letakkan berkas itu atau unggah CSV "
                 "di halaman Dataset.")
        return None, None, None

    try:
        mentah = baca_csv(isi)
    except Exception as e:
        st.error(f"CSV tidak dapat dibaca: {e}")
        return None, nama, None
    if mentah.empty:
        st.error("CSV tidak berisi baris data.")
        return None, nama, None

    # pemeriksaan kolom wajib
    cek = pd.DataFrame({'Kolom wajib': KOLOM_WAJIB,
                        'Status': ['ada' if k in mentah.columns else 'tidak ada' for k in KOLOM_WAJIB]})
    kurang = [k for k in KOLOM_WAJIB if k not in mentah.columns]
    if kurang:
        st.error(f"Kolom wajib tidak ditemukan: {kurang}. Aplikasi tidak dapat memproses data ini.")
        st.dataframe(cek, hide_index=True)
        return None, nama, cek

    df = mentah.copy()
    if 'tweet_url' in df.columns:
        df = df.drop(columns=['tweet_url'])
        st.session_state['tweet_url_dibuang'] = True
    else:
        st.session_state['tweet_url_dibuang'] = False
    return df, nama, cek


# ==========================================================
# HALAMAN: DATASET
# ==========================================================
def render_tab_data(df, nama, cek):
    if st.session_state.get('tweet_url_dibuang'):
        st.warning("Kolom 'tweet_url' memuat nama akun, jadi disembunyikan. Sebaiknya dibuang dari berkas sebelum dikumpulkan.")
    waktu = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    st.write("### Ringkasan Data")
    st.write(f"Jumlah data: {len(df):,} postingan · {df.shape[1]} kolom"
             + (f" · {waktu.min():%Y-%m-%d} s.d. {waktu.max():%Y-%m-%d}" if waktu.notna().any() else ""))
    st.caption(f"Sumber data: {nama}")

    with st.expander("Pemeriksaan kolom"):
        st.dataframe(cek, hide_index=True)
        opsional = {'text_clean': 'clustering & word cloud', 'sentimen': 'halaman Sentimen', 'cluster': 'halaman Clustering',
                    'retweet_count': 'regresi retweet & reply', 'reply_count': 'regresi retweet & reply',
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
        r = hasil_regresi(df)
        baris = [
            {'Analisis': 'Sentimen', 'Dipakai': 0 if d_s is None else n - k_s, 'Dikeluarkan': n if d_s is None else k_s,
             'Alasan': "kolom 'sentimen' tidak ada" if d_s is None else "label sentimen kosong/tidak valid"},
            {'Analisis': 'Clustering', 'Dipakai': 0 if d_c is None else n - k_c, 'Dikeluarkan': n if d_c is None else k_c,
             'Alasan': "kolom 'cluster' tidak ada" if d_c is None else "label cluster kosong"},
            {'Analisis': 'Regresi', 'Dipakai': r.get('n_pakai', 0),
             'Dikeluarkan': r['n_like_tidak_valid'] + r['n_teks_kosong'],
             'Alasan': f"likes kosong/bukan angka/negatif ({r['n_like_tidak_valid']}) dan teks kosong ({r['n_teks_kosong']}); likes tidak diganti 0"},
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
    return df


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
        st.warning("Berkas datas/evaluasi_sentimen.csv tidak ditemukan.")
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
        st.warning("Stopwords NLTK tidak tersedia (tidak ada internet?). Kata dominan dan peta bisa sedikit berbeda dari notebook. "
                   "Letakkan stopwords.txt di folder datas/ sebagai cadangan.")
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
def gambar_heatmap(data_latih):
    nama = {'jumlah_kata': 'Jumlah kata', 'jumlah_hashtag': 'Jumlah hashtag', 'like_count': 'Jumlah likes',
            'retweet_count': 'Jumlah retweet', 'reply_count': 'Jumlah reply'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    gambar = None
    for ax, (metode, judul) in zip(axes, [('pearson', 'Pearson (hubungan linear)'),
                                          ('spearman', 'Spearman (hubungan peringkat)')]):
        kor = data_latih.corr(method=metode).rename(index=nama, columns=nama)
        gambar = ax.imshow(kor.values, cmap='RdBu_r', vmin=-1, vmax=1)
        n = len(kor)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(kor.columns, rotation=20, ha='right')
        ax.set_yticklabels(kor.index)
        for i in range(n):
            for j in range(n):
                v = kor.values[i, j]
                ax.text(j, i, 'n/a' if np.isnan(v) else f'{v:.2f}', ha='center', va='center', fontsize=11,
                        color='white' if (not np.isnan(v) and abs(v) > 0.6) else '#2c2c2a')
        ax.set_title(judul, fontsize=11)
        ax.grid(False)
    fig.colorbar(gambar, ax=axes, shrink=0.85, label='Koefisien korelasi')
    fig.suptitle(f'Korelasi fitur dan likes (data latih, n = {len(data_latih)})', fontsize=12)
    return fig


def render_tab_regresi(df):
    st.subheader("Regresi Linear: Jumlah Kata dan Hashtag → Jumlah Likes")
    r = hasil_regresi(df)
    if not r['ok']:
        st.error(r['pesan'])
        return

    st.caption(f"Fitur dihitung dari teks asli. Data latih {r['n_latih']} baris (80%), data uji {r['n_uji']} baris (20%), "
               f"random_state = {RANDOM_STATE}. Model hanya belajar dari data latih; semua angka di bawah dari data uji.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{r['mae']:.2f} likes")
    m2.metric("R²", f"{r['r2']:.3f}")
    m3.metric("MAE tebak rata-rata", f"{r['mae_rata']:.2f}")
    m4.metric("MAE tebak median", f"{r['mae_median']:.2f}")
    st.code(f"likes ≈ {r['intercept']:.2f} + ({r['coef'][0]:.3f} × jumlah_kata) + ({r['coef'][1]:.3f} × jumlah_hashtag)",
            language=None)
    with st.expander("Statistik fitur dan sebaran hashtag"):
        st.dataframe(r['deskripsi'], width='stretch')
        st.write("**Sebaran jumlah hashtag (seluruh data valid):**")
        st.dataframe(r['sebaran_hashtag'].rename_axis('Jumlah hashtag').reset_index(name='Jumlah postingan'),
                     hide_index=True)
        st.caption(f"Postingan dengan ≥ 1 hashtag di data latih: {r['n_hashtag_latih']} dari {r['n_latih']}. "
                   "Fitur hashtag hampir konstan (kebanyakan bernilai 0), sehingga hanya sedikit membantu prediksi.")

    # grafik aktual vs prediksi
    st.write("### Likes Aktual vs Prediksi (Data Uji)")
    kiri, kanan = st.columns([1.1, 1])
    with kiri:
        log_skala = st.checkbox("Skala logaritmik (symlog), agar titik tidak menumpuk", value=False)
        y_te, y_pr = r['y_test'], r['y_pred']
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(y_te, y_pr, s=22, alpha=0.7, color='#2a78d6', edgecolors='white', linewidths=0.5,
                   label='Postingan (data uji)')
        batas = [min(y_te.min(), y_pr.min()), max(y_te.max(), y_pr.max())]
        ax.plot(batas, batas, linestyle='--', linewidth=1.5, color='#52514e', label='Prediksi sempurna (aktual = prediksi)')
        ax.axhline(0, color='#e34948', linewidth=1, alpha=0.6, label='Likes = 0 (di bawahnya = prediksi negatif)')
        if log_skala:
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
        ax.set_xlabel('Likes aktual')
        ax.set_ylabel('Likes prediksi')
        ax.set_title('Likes Aktual vs Prediksi (data uji)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with kanan:
        tabel = r['X_test'].copy()
        tabel['like_aktual'] = r['y_test']
        tabel['like_prediksi'] = r['y_pred'].round(2)
        st.write("**Tabel data uji:**")
        st.dataframe(tabel, width='stretch', height=380)

    # interpretasi
    st.write("### Interpretasi")
    st.markdown(
        f"- **MAE {r['mae']:.2f}**: rata-rata, perkiraan model meleset sekitar {r['mae']:.0f} likes dari jumlah likes sebenarnya "
        f"(satuan sama dengan likes). Median likes data uji hanya {r['median_uji']:.0f}, jadi angka ini perlu dibaca "
        f"terhadap sebaran data dan pembanding di atas.\n"
        f"- **R² {r['r2']:.3f}**: 1 berarti sempurna, 0 berarti sama baiknya dengan menebak rata-rata, dan **negatif berarti "
        f"lebih buruk daripada menebak rata-rata**.\n"
        f"- **Keterbatasan**: hanya dua fitur dan hanya {r['persen_berhashtag']:.1f}% postingan yang memakai hashtag. "
        f"Likes juga dipengaruhi usia postingan, jangkauan akun, dan isi pesan yang tidak ada dalam model."
    )
    if r['r2'] < 0:
        st.warning("R² negatif: pada data uji, model dua fitur ini tidak lebih baik daripada menebak rata-rata likes.")
    if r['mae_median'] < r['mae']:
        st.info(f"Menebak median likes (MAE {r['mae_median']:.2f}) lebih akurat daripada model ({r['mae']:.2f}). "
                "Sebaran likes sangat miring: sebagian besar postingan hanya mendapat sedikit likes.")
    st.caption(f"Prediksi negatif pada data uji: {r['n_negatif']} dari {r['n_uji']}. Likes negatif tidak bermakna "
               "dan menunjukkan keterbatasan regresi linear.")
    with st.expander("Lima postingan dengan selisih prediksi terbesar (data uji)"):
        sel = r['X_test'].copy()
        sel['aktual'] = r['y_test']
        sel['prediksi'] = r['y_pred'].round(1)
        sel['selisih'] = (sel['aktual'] - sel['prediksi']).abs()
        sel['text'] = [t[:200].replace('\n', ' ') for t in r['teks_uji']]
        st.dataframe(sel.sort_values('selisih', ascending=False).head(5), hide_index=True, width='stretch')
        st.caption("Selisih terbesar umumnya berasal dari postingan dengan likes sangat tinggi, yang tidak dapat "
                   "dijelaskan oleh jumlah kata dan hashtag.")

    with st.expander("Heatmap korelasi fitur dan likes (data latih)"):
        fig = gambar_heatmap(r['data_latih'])
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Pearson mengukur hubungan linear dengan nilai asli. Spearman memakai peringkat sehingga lebih tahan "
                   "terhadap postingan viral. Korelasi tidak menunjukkan sebab-akibat.")

    if r['eks'] is not None:
        st.info(f"Model tambahan **retweet + reply → likes** (MAE {r['eks']['mae']:.2f}, R² {r['eks']['r2']:.3f}) "
                "tersedia di menu sidebar **Regresi: Retweet & Reply**.")

    # formulir estimasi (model utama dua fitur)
    st.write("### Estimasi Likes")
    with st.form("form_estimasi"):
        f1, f2 = st.columns(2)
        w_in = f1.number_input("Jumlah kata", min_value=1, max_value=300,
                               value=int(min(max(r['kata_median'], 1), 300)), step=1)
        h_in = f2.number_input("Jumlah hashtag", min_value=0, max_value=50, value=0, step=1)
        kirim = st.form_submit_button("Hitung estimasi")
    if kirim:
        est = float(r['model'].predict(pd.DataFrame([[w_in, h_in]], columns=FITUR))[0])
        st.metric("Estimasi likes (nilai mentah model)", f"{est:,.1f}")
        if est < 0:
            st.warning("Estimasi negatif tidak bermakna sebagai jumlah likes. Ini menunjukkan keterbatasan model linear dua fitur.")
        if w_in > r['kata_max'] or h_in > r['hashtag_max']:
            st.warning(f"Masukan di luar rentang data (maksimum {r['kata_max']} kata dan {r['hashtag_max']} hashtag); "
                       "estimasi tidak andal.")
        st.caption(f"Sebagai pembanding: median likes seluruh data {r['median_semua']:.0f}, sedangkan model memperkirakan "
                   f"sekitar {r['est_tipikal']:.0f} likes untuk postingan tipikal ({r['kata_median']} kata, tanpa hashtag). "
                   "Estimasi ini hanya ilustrasi, bukan prediksi yang dapat diandalkan.")


# ==========================================================
# HALAMAN: REGRESI RETWEET + REPLY (model tambahan notebook)
# ==========================================================
def render_tab_regresi_engagement(df):
    st.subheader("Regresi Linear: Jumlah Retweet dan Reply → Jumlah Likes")
    r = hasil_regresi(df)
    if not r['ok']:
        st.error(r['pesan'])
        return
    e = r['eks']
    if e is None:
        st.error("Kolom 'retweet_count' dan 'reply_count' tidak ada atau berisi nilai kosong/bukan angka, "
                 "sehingga model ini tidak dapat dilatih.")
        return

    st.caption(f"Sama dengan model tambahan (non-log) di notebook: skala asli, baris data latih ({r['n_latih']}) dan "
               f"data uji ({r['n_uji']}) identik dengan model jumlah kata & hashtag (80/20, random_state = {RANDOM_STATE}). "
               "Model hanya belajar dari data latih; semua angka di bawah dari data uji.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{e['mae']:.2f} likes")
    m2.metric("R²", f"{e['r2']:.3f}")
    m3.metric("MAE tebak rata-rata", f"{r['mae_rata']:.2f}")
    m4.metric("MAE tebak median", f"{r['mae_median']:.2f}")
    st.code(f"likes ≈ {e['intercept']:.2f} + ({e['coef'][0]:.3f} × retweet) + ({e['coef'][1]:.3f} × reply)",
            language=None)

    st.write("### Perbandingan dengan Model Jumlah Kata & Hashtag")
    st.dataframe(pd.DataFrame({
        'Model': ['Jumlah kata + hashtag', 'Retweet + reply', 'Tebak rata-rata', 'Tebak median'],
        'MAE (likes)': [round(r['mae'], 2), round(e['mae'], 2), round(r['mae_rata'], 2), round(r['mae_median'], 2)],
        'R²': [round(r['r2'], 3), round(e['r2'], 3), None, None],
        'Prediksi negatif': [r['n_negatif'], e['n_negatif'], None, None],
    }), hide_index=True, width='stretch')

    with st.expander("Statistik fitur retweet, reply, dan likes"):
        st.dataframe(e['deskripsi'], width='stretch')

    # grafik aktual vs prediksi
    st.write("### Likes Aktual vs Prediksi (Data Uji)")
    kiri, kanan = st.columns([1.1, 1])
    with kiri:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        ax.scatter(r['y_test'], e['y_pred'], s=22, alpha=0.7, color='#eb6834',
                   edgecolors='white', linewidths=0.5, label='Postingan (data uji)')
        batas = max(r['y_test'].max(), e['y_pred'].max())
        ax.plot([0, batas], [0, batas], '--', color='#52514e', label='Prediksi sempurna')
        ax.set_xscale('symlog'); ax.set_yscale('symlog')
        ax.set_xlabel('Likes aktual'); ax.set_ylabel('Likes prediksi')
        ax.set_title('Retweet + reply → likes (skala symlog)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with kanan:
        tabel = e['X_test'].copy()
        tabel['like_aktual'] = r['y_test']
        tabel['like_prediksi'] = e['y_pred'].round(2)
        st.write("**Tabel data uji:**")
        st.dataframe(tabel, width='stretch', height=380)

    # interpretasi
    st.write("### Interpretasi")
    catatan = f"Prediksi negatif: {e['n_negatif']} dari {r['n_uji']}."
    if e['intercept'] > r['median_uji'] and min(e['coef']) >= 0:
        catatan += (f" Karena koefisiennya tidak negatif, prediksi tidak pernah di bawah ±{e['intercept']:.1f} likes, "
                    f"padahal median likes data uji {r['median_uji']:.0f}; postingan yang sepi cenderung "
                    "diprediksi terlalu tinggi.")
    banding = "lebih kecil" if e['mae'] < r['mae_median'] else "tidak lebih kecil"
    arti_r2 = ("sebagian besar variasi likes di data uji ikut dijelaskan oleh retweet dan reply" if e['r2'] >= 0.5
               else "model lebih buruk daripada menebak rata-rata" if e['r2'] < 0
               else "hanya sebagian variasi likes yang dijelaskan oleh retweet dan reply")
    st.markdown(
        f"- **MAE {e['mae']:.2f}**: rata-rata perkiraan meleset sekitar {e['mae']:.0f} likes, {banding} daripada "
        f"menebak median ({r['mae_median']:.2f}); model jumlah kata & hashtag MAE {r['mae']:.2f}.\n"
        f"- **R² {e['r2']:.3f}**: {arti_r2}.\n"
        f"- Setiap tambahan 1 retweet berasosiasi dengan ±{e['coef'][0]:.2f} likes, dan 1 reply dengan "
        f"±{e['coef'][1]:.2f} likes (dengan fitur lain tetap)."
    )
    st.caption(catatan)
    st.warning("Retweet dan reply baru diketahui setelah postingan mendapat respons, sama seperti likes. "
               "Model ini menggambarkan hubungan antar-metrik engagement, bukan prediksi likes sebelum postingan terbit.")

    with st.expander("Heatmap korelasi retweet, reply, dan likes (data latih)"):
        fig = gambar_heatmap(e['data_latih'])
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Korelasi tidak menunjukkan sebab-akibat.")

    # formulir estimasi
    st.write("### Estimasi Likes dari Retweet dan Reply")
    with st.form("form_estimasi_engagement"):
        f1, f2 = st.columns(2)
        rt_in = f1.number_input("Jumlah retweet", min_value=0, max_value=100000,
                                value=int(e['median']['retweet_count']), step=1)
        rp_in = f2.number_input("Jumlah reply", min_value=0, max_value=10000,
                                value=int(e['median']['reply_count']), step=1)
        kirim = st.form_submit_button("Hitung estimasi")
    if kirim:
        est = float(e['model'].predict(pd.DataFrame([[rt_in, rp_in]], columns=FITUR_ENG))[0])
        st.metric("Estimasi likes (nilai mentah model)", f"{est:,.1f}")
        if est < 0:
            st.warning("Estimasi negatif tidak bermakna sebagai jumlah likes.")
        if rt_in > e['maks']['retweet_count'] or rp_in > e['maks']['reply_count']:
            st.warning(f"Masukan di luar rentang data (maksimum {e['maks']['retweet_count']} retweet dan "
                       f"{e['maks']['reply_count']} reply); estimasi tidak andal.")
        st.caption(f"Pembanding: median likes seluruh data {r['median_semua']:.0f}. Estimasi ini hanya ilustrasi.")


# ==========================================================
# TEMUAN DAN KETERBATASAN
# ==========================================================
def render_temuan(df):
    st.subheader("Temuan Utama dan Keterbatasan")
    temuan = []

    d, _ = siapkan_sentimen(df)
    if d is not None and len(d):
        p = (d['sentimen'].value_counts(normalize=True) * 100).reindex(URUTAN_SENTIMEN, fill_value=0)
        urut = p.sort_values(ascending=False)
        kal = (f"**Sentimen.** Dari {len(d)} postingan, kategori terbanyak adalah *{urut.index[0]}* "
               f"({urut.iloc[0]:.1f}%). Sebaran: positif {p['positif']:.1f}%, netral {p['netral']:.1f}%, "
               f"negatif {p['negatif']:.1f}%.")
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

    r = hasil_regresi(df)
    if r['ok']:
        arah = ("model lebih buruk daripada menebak rata-rata" if r['r2'] < 0
                else "model hanya menjelaskan sebagian kecil variasi likes")
        kal_reg = (f"**Regresi.** Dengan jumlah kata dan hashtag, MAE {r['mae']:.1f} likes dan R² {r['r2']:.3f} "
                   f"({arah}). Menebak median likes menghasilkan MAE {r['mae_median']:.1f}.")
        if r['eks'] is not None:
            kal_reg += (f" Pada analisis tambahan, retweet dan reply (terukur bersamaan dengan likes) memberi MAE "
                        f"{r['eks']['mae']:.1f} dan R² {r['eks']['r2']:.3f}; ini hubungan antar-metrik engagement, "
                        "bukan prediksi sebelum posting.")
        temuan.append(kal_reg)

    if temuan:
        st.write("### 3 Temuan Utama")
        for i, t in enumerate(temuan, 1):
            st.markdown(f"{i}. {t}")
    else:
        st.info("Temuan belum dapat dihitung karena kolom analisis belum tersedia pada data.")

    waktu = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    periode = f"{waktu.min():%Y-%m-%d} s.d. {waktu.max():%Y-%m-%d}" if waktu.notna().any() else "periode tidak diketahui"
    hashtag_info = f" Hanya {r['persen_berhashtag']:.1f}% postingan yang memakai hashtag." if r['ok'] else ""
    st.write("### 2 Keterbatasan")
    st.markdown(
        f"1. **Data sampel.** Data berasal dari pencarian kata kunci tertentu ({periode}), bukan seluruh percakapan di X, "
        "sehingga tidak mewakili opini masyarakat. Sentimen diberi label oleh model pra-latih dan hanya diperiksa manual pada "
        "sejumlah kecil sampel; sebagian postingan tidak relevan atau berbahasa lain masih mungkin ada.\n"
        "2. **Model likes sederhana.** Likes dipengaruhi usia postingan, jangkauan akun, dan isi pesan yang tidak ada "
        f"dalam model, dan sebaran likes sangat miring.{hashtag_info}"
    )


# ==========================================================
# HALAMAN: TENTANG
# ==========================================================
def render_tentang():
    st.subheader("Tentang")
    st.write(
        "Aplikasi ini menampilkan hasil analisis postingan X tentang AI dan profesi kreatif. "
        "Alur: crawling, pembersihan data, klasifikasi sentimen (IndoBERT), clustering TF-IDF + K-Means (K = 3), "
        "dan regresi linear untuk jumlah likes. Proses berat dijalankan di notebook; aplikasi membaca hasilnya "
        "dari folder datas/."
    )
    st.write("### Informasi Pengambilan Data")
    st.dataframe(pd.DataFrame(list(INFO_PENGAMBILAN.items()), columns=['Item', 'Keterangan']),
                 hide_index=True, width='stretch')


# ==========================================================
# TATA LETAK UTAMA
# ==========================================================
MENU = ["Dataset", "Sentimen", "Clustering", "Regresi: Kata & Hashtag",
        "Regresi: Retweet & Reply", "Temuan & Keterbatasan", "Tentang"]
HALAMAN = {
    "Sentimen": render_tab_sentimen,
    "Clustering": render_tab_cluster,
    "Regresi: Kata & Hashtag": render_tab_regresi,
    "Regresi: Retweet & Reply": render_tab_regresi_engagement,
    "Temuan & Keterbatasan": render_temuan,
}


def main():
    st.title("Dashboard Analisis AI pada Profesi Kreatif")
    st.caption("Analisis sentimen, clustering, dan regresi linear pada postingan X")

    choice = st.sidebar.selectbox("Menu", MENU, key='menu')

    if choice == "Tentang":
        render_tentang()
        return
    if choice == "Dataset":
        st.subheader("Dataset")
        widget_unggah()

    data, nama_berkas, tabel_cek = muat_data()
    if data is None:
        st.info("Muat data yang valid terlebih dahulu di halaman Dataset.")
    elif choice == "Dataset":
        render_tab_data(data, nama_berkas, tabel_cek)
    else:
        HALAMAN[choice](data)


if __name__ == "__main__":
    main()