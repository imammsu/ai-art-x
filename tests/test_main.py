"""Pengujian untuk main.py.

Jalankan:  .\\venv\\Scripts\\python.exe -m pytest -v
"""
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / 'main.py')
sys.path.insert(0, str(ROOT))


# ----------------------------------------------------------
# Impor modul (Streamlit berjalan dalam "bare mode")
# ----------------------------------------------------------
@pytest.fixture(scope='session')
def m():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return importlib.import_module('main')


@pytest.fixture(scope='session')
def df_asli(m):
    return m.baca_csv(m.PATH_DATA.read_bytes())


# ==========================================================
# UNIT TEST: fungsi bantu
# ==========================================================
def test_berkas_data_ada(m):
    for p in (m.PATH_DATA, m.PATH_RINGKASAN, m.PATH_EVAL, m.PATH_STOPWORDS, m.PATH_TAHAP):
        assert p.exists(), f"{p} tidak ditemukan"


def test_baca_csv_koma_dan_titik_koma(m):
    a = m.baca_csv(b"a,b\n1,2\n")
    b = m.baca_csv(b"a;b\n1;2\n")
    c = m.baca_csv("\ufeffa,b\n1,2\n".encode('utf-8'))   # BOM
    for d in (a, b, c):
        assert list(d.columns) == ['a', 'b']
        assert d.iloc[0].tolist() == [1, 2]


def test_dataset_punya_kolom_wajib(m, df_asli):
    assert len(df_asli) > 0
    assert set(m.KOLOM_WAJIB).issubset(df_asli.columns)


def test_muat_stopwords(m):
    sw, ok = m.muat_stopwords()
    assert ok
    assert 'ai' in sw and 'yang' in sw
    assert sw == sorted(set(sw))


@pytest.mark.parametrize('k,n', [(0, 10), (5, 10), (10, 10), (37, 50)])
def test_wilson_dalam_rentang(m, k, n):
    lo, hi = m.wilson(k, n)
    assert 0 <= lo <= k / n <= hi <= 1


def test_wilson_n_nol(m):
    assert m.wilson(0, 0) == (0.0, 0.0)


def test_siapkan_sentimen(m):
    df = pd.DataFrame({'sentimen': [' Positif', 'netral', 'NEGATIF', None, 'x']})
    d, keluar = m.siapkan_sentimen(df)
    assert keluar == 2
    assert d['sentimen'].tolist() == ['positif', 'netral', 'negatif']
    assert m.siapkan_sentimen(pd.DataFrame({'a': [1]})) == (None, 0)


def test_siapkan_cluster(m):
    df = pd.DataFrame({'cluster': [0, '1', None, 2, 7]})
    d, keluar = m.siapkan_cluster(df)
    assert keluar == 1
    assert d['cluster'].tolist() == [0, 1, 2, 7]
    assert d['tema'].iloc[0] == m.NAMA_TEMA[0]
    assert d['tema'].iloc[-1] == 'Cluster 7'     # cluster tanpa nama tema
    assert m.siapkan_cluster(pd.DataFrame({'a': [1]})) == (None, 0)


def test_kesesuaian_query(m):
    dc = pd.DataFrame({'query': ['a', 'a', 'b', 'b'], 'cluster': [0, 0, 1, 1]})
    assert m.kesesuaian_query(dc) == pytest.approx(100.0)
    dc2 = pd.DataFrame({'query': ['a', 'a', 'b', 'b'], 'cluster': [0, 1, 0, 1]})
    assert m.kesesuaian_query(dc2) == pytest.approx(50.0)
    assert m.kesesuaian_query(pd.DataFrame({'cluster': [0]})) is None


def test_ringkas_evaluasi(m):
    ev = m.ringkas_evaluasi()
    assert ev is not None
    assert 0 <= ev['k'] <= ev['n']
    assert ev['kelas'] in m.URUTAN_SENTIMEN


# ==========================================================
# UNIT TEST: analisis
# ==========================================================
def test_latih_regresi_data_asli(m, df_asli):
    r = m.hasil_regresi(df_asli)
    assert r['ok'], r.get('pesan')
    assert r['n_latih'] + r['n_uji'] == r['n_pakai']
    assert r['n_uji'] == pytest.approx(r['n_pakai'] * 0.2, abs=1)
    assert r['mae'] >= 0 and np.isfinite(r['r2'])
    assert len(r['coef']) == 2
    assert len(r['y_pred']) == r['n_uji'] == len(r['teks_uji'])


def test_latih_regresi_deterministik(m, df_asli):
    m.latih_regresi.clear()
    r1 = m.hasil_regresi(df_asli)
    m.latih_regresi.clear()
    r2 = m.hasil_regresi(df_asli)
    assert r1['mae'] == r2['mae'] and r1['coef'] == r2['coef']


def test_latih_regresi_buang_like_tidak_valid(m):
    n = 30
    df = pd.DataFrame({'text': [f'kata {i} #tag' if i % 3 == 0 else f'kata kata {i}' for i in range(n)],
                       'like_count': list(range(n))})
    df.loc[0, 'like_count'] = -5
    df.loc[1, 'like_count'] = None
    df.loc[2, 'text'] = '   '
    r = m.latih_regresi(df)
    assert r['ok']
    assert r['n_like_tidak_valid'] == 2
    assert r['n_teks_kosong'] == 1
    assert r['n_pakai'] == n - 3
    assert r['eks'] is None                       # tanpa retweet/reply


def test_latih_regresi_data_terlalu_sedikit(m):
    df = pd.DataFrame({'text': ['a b'] * 5, 'like_count': [1] * 5})
    r = m.latih_regresi(df)
    assert not r['ok'] and 'minimal 20' in r['pesan']


def test_analisis_cluster_data_asli(m, df_asli):
    d, _ = m.siapkan_cluster(df_asli)
    d = d.reset_index(drop=True)
    sw, _ = m.muat_stopwords()
    import re
    stop = tuple(sorted(w for w in sw if re.fullmatch(r'[a-z]{3,}', w)))
    koordinat, info, sil, tanpa = m.analisis_cluster(
        tuple(d['text_clean'].fillna('').astype(str)), tuple(d['cluster']), stop)
    assert koordinat.shape == (len(d), 2)
    assert set(info) == set(d['cluster'].unique())
    assert -1 <= sil <= 1
    for c, v in info.items():
        assert len(v['top5']) == 5 and len(v['contoh']) <= 3
        assert all(d.loc[i, 'cluster'] == c for i in v['contoh'])


def test_buat_wordcloud(m):
    teks = ('gambar bagus sekali', 'jelek buruk sekali', 'biasa saja')
    hasil = m.buat_wordcloud(teks, ('positif', 'negatif', 'positif'), ('saja',))
    assert hasil['netral'] is None
    assert hasil['positif'].ndim == 3 and hasil['negatif'].ndim == 3
# ==========================================================
# KECOCOKAN DENGAN NOTEBOOK (cell 54 & 63 / parameter_model.json)
# ==========================================================
def test_model_sama_dengan_notebook(m, df_asli):
    import json
    p = json.loads((ROOT / 'parameter_model.json').read_text(encoding='utf-8'))
    r = m.hasil_regresi(df_asli)
    u, t = p['model_utama'], p['model_tambahan']
    assert r['mae'] == pytest.approx(u['mae'], abs=1e-6)
    assert r['r2'] == pytest.approx(u['r2'], abs=1e-6)
    assert r['coef'] == pytest.approx(u['koefisien'], abs=1e-6)
    e = r['eks']
    assert e is not None
    assert e['mae'] == pytest.approx(t['mae'], abs=1e-6)          # 48.30
    assert e['r2'] == pytest.approx(t['r2'], abs=1e-6)            # 0.673
    assert e['intercept'] == pytest.approx(t['intercept'], abs=1e-6)
    assert e['coef'] == pytest.approx(t['koefisien'], abs=1e-6)
    assert e['maks'] == t['rentang']


# ==========================================================
# END-TO-END: aplikasi Streamlit (AppTest)
# ==========================================================
MENU = ["📁 Data", "💬 Sentimen", "🧩 Clustering", "📈 Regresi: Kata & Hashtag",
        "📊 Regresi: Retweet & Reply", "📝 Temuan & Keterbatasan"]


def buka(menu):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio(key='menu').set_value(menu).run()
    return at


def test_sidebar_menu():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 0                                     # tidak lagi memakai tab
    radio = at.sidebar.radio(key='menu')
    assert list(radio.options) == MENU
    assert radio.value == "📁 Data"


@pytest.mark.parametrize('menu', MENU)
def test_setiap_halaman_tanpa_error(menu):
    at = buka(menu)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_halaman_data_filter():
    at = buka("📁 Data")
    assert 'Jumlah postingan' in [mt.label for mt in at.metric]
    at.text_input[0].input('hak cipta').run()
    assert not at.exception
    assert any(c.value.startswith('Menampilkan') for c in at.caption)


def test_halaman_sentimen_filter():
    at = buka("💬 Sentimen")
    assert 'Sesuai' in [mt.label for mt in at.metric]
    for v in ['positif', 'netral', 'negatif', 'Semua']:
        at.selectbox(key='filter_tabel_sentimen').select(v).run()
        assert not at.exception


def test_halaman_cluster():
    at = buka("🧩 Clustering")
    assert 'Silhouette score (cosine)' in [mt.label for mt in at.metric]


def test_halaman_regresi_kata():
    at = buka("📈 Regresi: Kata & Hashtag")
    assert 'R²' in [mt.label for mt in at.metric]
    at.checkbox[0].check().run()
    assert not at.exception
    at.number_input[0].set_value(300)
    at.number_input[1].set_value(50)
    at.button[0].click().run()
    assert any(mt.label.startswith('Estimasi likes') for mt in at.metric)
    assert any('di luar rentang' in w.value for w in at.warning)


def test_halaman_regresi_retweet_reply():
    at = buka("📊 Regresi: Retweet & Reply")
    nilai = {mt.label: mt.value for mt in at.metric}
    assert nilai['MAE'] == '48.30 likes'
    assert nilai['R²'] == '0.673'
    assert any('4.959 × retweet' in c.value and '8.267 × reply' in c.value for c in at.code)
    at.number_input[0].set_value(100)
    at.number_input[1].set_value(10)
    at.button[0].click().run()
    assert not at.exception
    est = [mt.value for mt in at.metric if mt.label.startswith('Estimasi likes')]
    assert est == [f"{14.599969 + 4.959438 * 100 + 8.266765 * 10:,.1f}"]


def test_halaman_temuan():
    at = buka("📝 Temuan & Keterbatasan")
    teks = ' '.join(md.value for md in at.markdown)
    assert '**Sentimen.**' in teks and '**Clustering.**' in teks and '**Regresi.**' in teks
