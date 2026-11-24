# Indonesian Speech-to-Text

Aplikasi berbasis web untuk **Transkripsi Wicara Otomatis (*Speech-to-Text*)** dan **Peringkasan Eksekutif (*Hybrid Extractive-Abstractive Summarization*)** khusus percakapan dan narasi berbahasa Indonesia.

Sistem diadaptasi dari penelitian dan eksperimen pemodelan pada notebook: `Salinan_dari_NLP_FIX.ipynb`.

---

## Fitur Utama

1. **Multi-Format Input:**
   - **Unggah Berkas Audio:** Mendukung format `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm` dengan pemutar audio terintegrasi.
   - **Tautan Video YouTube:** Otomatis mengekstrak stream audio YouTube via `yt-dlp` tanpa konversi ulang berlebih.
   - **Input Teks Langsung:** Masukkan teks transkrip manual untuk langsung dibersihkan dan diringkas.
   - **Audio Sampel Evaluasi:** Berkas audio narasi 45 detik untuk demonstrasi dan evaluasi cepat pipeline.

2. **Mesin Speech-to-Text (Faster-Whisper):**
   - Didukung oleh `faster-whisper` (CTranslate2 INT8) dengan pilihan model **Whisper Small** dan **Whisper Medium**.
   - Dilengkapi *Voice Activity Detection* (VAD filter) dengan padding untuk memangkas jeda hening tanpa memotong artikulasi kata.
   - Optimasi decoding CPU hybrid (`beam_size=1`, `best_of=1`, `cpu_threads=6`, `condition_on_previous_text=True`).

3. **Pembersihan Kata Pengisi (Filler Words) & Ortografi:**
   - Membersihkan lebih dari 15 pola kata pengisi non-formal khas Indonesia (`ee`, `umm`, `eh`, `anu`, `hmm`, `ya kan`, `gitu loh`, `maksudnya`, dll.).
   - Menormalisasi perulangan vokal informal (*elongated vowels*, contoh: `bangeeet` -> `banget`).

4. **Peringkasan Teks Hybrid (IndoNanoT5-base):**
   - Menggunakan model Transformer Sequence-to-Sequence **`LazarusNLP/IndoNanoT5-base`**.
   - Menggabungkan ringkasan abstraktif T5 dengan kalimat kunci ekstraktif representatif dan deduplikasi overlap leksikal `< 50%`.
   - Inferensi teroptimasi CPU: `torch.inference_mode()`, batched chunk generation (`batch_size=4`), dan `repetition_penalty=1.5`.
   - Fleksibilitas panjang target: **Singkat** (~60-130 kata), **Standar** (~80-250 kata), atau **Terperinci** (~150-260 kata).

5. **Manajemen Memori RAM & Caching:**
   - *Lazy loading* bertahap: model tidak dimuat sebelum dipanggil.
   - *Memory guard* adaptif untuk mencegah akumulasi beban RAM saat pergantian model.
   - *Result caching* berbasis hash SHA-256 untuk pemrosesan instan pada masukan identik.

---

## Struktur Direktori

```text
d:/STT/
├── app.py                      # Antarmuka web utama berbasis Streamlit
├── core/
│   ├── config.py               # Konfigurasi model, parameter CPU, dan direktori
│   ├── transcriber.py          # Modul Speech-to-Text Faster-Whisper
│   ├── preprocessor.py         # Modul pembersihan filler words & ekstraksi kalimat kunci
│   ├── summarizer.py           # Modul summarization hybrid IndoNanoT5-base
│   ├── audio_downloader.py     # Modul ekstraksi audio YouTube via yt-dlp
│   ├── pipeline.py             # Orkestrasi pipeline pemrosesan dan manajemen cache
│   ├── memory_utils.py         # Pemantauan RAM dan pelepasan objek sementara
│   └── result_cache.py         # Manajemen cache berbasis hash SHA-256
├── utils/
│   └── export_helpers.py       # Helper ekspor hasil ke format TXT dan JSON
├── demo_data/
│   └── contoh_podcast_45s.mp3  # Audio sampel evaluasi (45 detik)
├── assets/
│   └── styles.css              # Styling antarmuka profesional minimalis (Inter font)
├── requirements.txt            # Daftar dependensi Python
├── run_tests.py                # Script pengujian dan validasi otomatis (Test 1-6)
└── README.md                   # Dokumentasi teknis sistem
```

---

## Panduan Instalasi dan Menjalankan Aplikasi

### 1. Prasyarat Sistem
- **Python:** Versi 3.10, 3.11, atau 3.12.
- **RAM:** Minimal 4 GB (Disarankan 8–16 GB).
- **Perangkat Keras:** Kompatibel penuh dengan CPU (Intel Core i7-1355U / Intel Iris Xe) atau GPU NVIDIA CUDA jika tersedia.

### 2. Instalasi Dependensi
Jalankan di terminal proyek:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
```bash
python -m streamlit run app.py
```
Aplikasi dapat diakses melalui peramban pada alamat lokal: `http://localhost:8501`.

---

## Struktur Halaman Aplikasi

Antarmuka dirancang dengan dua tab navigasi utama:
1. **Pemrosesan Transkripsi:**
   - Tiga metode masukan: Unggah Berkas Audio, URL YouTube, dan Teks Langsung.
   - Pilihan model Faster-Whisper Small / Medium, pembersihan kata pengisi, dan panjang ringkasan.
   - Hasil terstruktur: Sumber, Model, Durasi & Waktu Transkripsi, RTF & Waktu Peringkasan, Ringkasan Eksekutif, Tab Transkrip Bersih & Mentah, serta Tombol Unduh TXT dan JSON.
2. **Evaluasi Model:**
   - Perbandingan karakteristik arsitektur Whisper Small vs Medium.
   - Metrik akurasi ground truth (WER ~16.5%, CER ~6.8%, ROUGE-1/2/L).
   - Grafik interaktif Altair untuk Real-Time Factor (RTF) dan faktor percepatan.
   - Tabel pengujian benchmark empiris bertingkat (45s, 4m, 12m, 25.6m).
   - Analisis penggunaan RAM dan proteksi memori sistem.

---

## Hasil Evaluasi Model

### 1. Metrik Ground Truth Penelitian (Dataset Podcast Satu Persen)
| Metrik Evaluasi | Nilai Rata-Rata | Kategori |
| :--- | :--- | :--- |
| **Word Error Rate (WER)** | **~16.5%** | Akurasi Baik pada narasi santai |
| **Character Error Rate (CER)** | **~6.8%** | Akurasi karakter sangat baik |
| **ROUGE-1 F1** | **0.528** | Presisi unigram tinggi |
| **ROUGE-2 F1** | **0.466** | Koherensi bigram terjaga |
| **ROUGE-L F1** | **0.469** | Struktur urutan kalimat konsisten |

### 2. Hasil Pengujian Benchmark Empiris (Intel Core i7-1355U, RAM 16 GB)
| Durasi Audio | Model | Waktu Transkripsi (Sebelum) | Waktu Transkripsi (Sesudah) | RTF Sebelum | RTF Sesudah | Waktu Peringkasan (Sebelum) | Waktu Peringkasan (Sesudah) | Faktor Percepatan Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **45 Detik** | Small | 19.31 s | **8.84 s** | 0.429 | **0.196** | 33.52 s | **13.79 s** | **2.18x** |
| **45 Detik** | Medium | 102.05 s | **37.87 s** | 2.268 | **0.842** | 97.56 s | **14.19 s** | **3.85x** |
| **4 Menit** | Small | 191.68 s | **78.44 s** | 0.799 | **0.327** | 113.48 s | **25.02 s** | **2.93x** |
| **12 Menit** | Small | 559.84 s | **323.00 s** | 0.778 | **0.449** | 274.05 s | **95.99 s** | **1.96x** |
| **25.6 Menit** | Small | ~1,200 s | **644.44 s** | ~0.780 | **0.420** | ~550 s | **99.73 s** | **~2.33x** |
