========================================================================
 DASHBOARD ANALISIS AI PADA PROFESI KREATIF - APLIKASI WEB STREAMLIT
========================================================================

Aplikasi web interaktif untuk analisis sentimen, clustering teks, dan estimasi
regresi linear terkait opini profesi kreatif terhadap Artificial Intelligence.

------------------------------------------------------------------------
1. KEBUTUHAN SISTEM & DAFTAR LIBRARY (REQUIREMENTS)
------------------------------------------------------------------------
Python Versi: Python 3.10+ (Direkomendasikan Python 3.10 - 3.12)

Daftar library yang digunakan:
- streamlit           (Framework antarmuka web interaktif)
- pandas              (Manipulasi dan pemrosesan dataset CSV)
- numpy               (Komputasi numerik dan manipulasi matriks)
- matplotlib          (Visualisasi grafik dan diagram)
- seaborn             (Visualisasi statistik data)
- scikit-learn        (Algoritma ML: K-Means, TF-IDF, TruncatedSVD, Regresi Linear)
- joblib              (Pemuatan model machine learning .pkl)
- wordcloud           (Pembuatan visualisasi word cloud kata kunci)
- nltk                (Pengolahan stopwords bahasa Indonesia)

Perintah instalasi seluruh dependensi sekaligus:
  pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib wordcloud nltk


------------------------------------------------------------------------
2. CARA MENJALANKAN APLIKASI DENGAN VIRTUAL ENVIRONMENT
------------------------------------------------------------------------

Langkah-langkah menjalankan aplikasi secara lokal:

1. Buka Terminal / PowerShell / Command Prompt di dalam direktori repositori ini.

2. Buat dan aktifkan Virtual Environment:

   a. Membuat virtual environment (contoh nama: 'venv'):
      python -m venv venv

   b. Mengaktifkan virtual environment:
      - Windows (PowerShell):
        .\venv\Scripts\Activate.ps1
        (Jika muncul pesan error ExecutionPolicy, jalankan:
         Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process)

      - Windows (Command Prompt / CMD):
        venv\Scripts\activate.bat

      - macOS / Linux (Bash/Zsh):
        source venv/bin/activate

3. Perbarui pip dan pasang library dependensi:
      python -m pip install --upgrade pip
      pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib wordcloud nltk

4. Jalankan aplikasi Streamlit:
      streamlit run main.py

5. Buka tautan lokal pada web browser Anda:
      http://localhost:8501


------------------------------------------------------------------------
3. STRUKTUR REPOSITORI
------------------------------------------------------------------------
├── main.py                  # Skrip utama aplikasi Streamlit
├── datas/                   # Dataset dan hasil analisis CSV
│   ├── dataset_final.csv
│   ├── hasil_sentimen.csv
│   ├── hasil_cluster.csv
│   ├── hasil_regresi_data_uji.csv
│   └── evaluasi_manual.csv
├── model_regresi.pkl        # Model regresi linear yang telah dilatih
└── README.txt               # Dokumentasi instalasi dan penggunaan
========================================================================
