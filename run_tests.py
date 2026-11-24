import os
import sys
import json
import wave
import struct

# Tambahkan root workspace ke sys.path
sys.path.insert(0, r"d:\STT")

from core.preprocessor import clean_text_id, get_cleaning_statistics, extract_key_sentences
from core.audio_downloader import is_valid_youtube_url, download_youtube_audio, check_ffmpeg_availability
from core.config import FFMPEG_BIN_DIR, FFMPEG_PATH, FFPROBE_PATH
from core.summarizer import generate_hybrid_summary, chunk_text
from core.transcriber import transcribe_audio, get_audio_duration_seconds
from utils.export_helpers import create_export_txt, create_export_json

print("=" * 60)
print("MEMULAI PENGUJIAN DAN VALIDASI SISTEM (TAHAP 4)")
print("=" * 60)

# -------------------------------------------------------------
# TEST 1: Preprocessor & Text Cleaning
# -------------------------------------------------------------
print("\n[TEST 1] Pengujian Preprocessor & Text Cleaning...")
sample_raw = "halo semuanya ee apa kabar umm hari ini kita mau ngomongin kesehatan mental yang bangeeet penting ya kan gitu loh"
cleaned = clean_text_id(sample_raw)
stats = get_cleaning_statistics(sample_raw, cleaned)

print(f"Raw     : '{sample_raw}'")
print(f"Cleaned : '{cleaned}'")
print(f"Statistik: {stats}")

assert "ee" not in cleaned.split(), "Pola 'ee' harus terhapus!"
assert "umm" not in cleaned.split(), "Pola 'umm' harus terhapus!"
assert "banget" in cleaned, "Kata 'bangeeet' harus dinormalisasi menjadi 'banget'!"
assert stats["removed_filler_count"] > 0, "Filler count harus berkurang!"
print(">>> TEST 1 BERHASIL! (Pembersihan filler & normalisasi vokal akurat)")

# -------------------------------------------------------------
# TEST 2: Ketersediaan FFmpeg/FFprobe, Validasi URL YouTube & Unduhan
# -------------------------------------------------------------
print("\n[TEST 2] Pengujian Deteksi FFmpeg, FFprobe & Integrasi yt-dlp...")

# 2A. Verifikasi ketersediaan FFmpeg dan FFprobe
ready_ff, msg_ff, details_ff = check_ffmpeg_availability()
print(f"Status FFmpeg: ready={ready_ff}, bin_dir='{details_ff.get('bin_dir')}'")
print(f"  - ffmpeg executable: {details_ff.get('ffmpeg_path')}")
print(f"  - ffprobe executable: {details_ff.get('ffprobe_path')}")
assert details_ff.get("has_ffmpeg") == True, "Executable ffmpeg harus terdeteksi!"
assert details_ff.get("has_ffprobe") == True, "Executable ffprobe harus terdeteksi di folder bin!"

# Verifikasi langsung bahwa yt-dlp mengenali ffmpeg & ffprobe melalui FFmpegPostProcessor
import yt_dlp
test_ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'ffmpeg_location': details_ff.get('bin_dir')
}
with yt_dlp.YoutubeDL(test_ydl_opts) as ydl:
    pp = yt_dlp.postprocessor.FFmpegPostProcessor(ydl)
    versions, _ = pp.get_versions_and_features(None)
    print(f"Versi FFmpeg yang dideteksi oleh yt-dlp: {versions}")
    assert "ffmpeg" in versions, "yt-dlp harus berhasil mendeteksi versi ffmpeg!"
    assert "ffprobe" in versions, "yt-dlp harus berhasil mendeteksi versi ffprobe!"
print(">>> Deteksi biner FFmpeg & FFprobe oleh yt-dlp TERVERIFIKASI!")

# 2B. Pengujian Validasi URL YouTube
valid_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
invalid_url = "https://google.com/not-youtube"
empty_url = ""

assert is_valid_youtube_url(valid_url) == True, "URL YouTube valid harus terdeteksi!"
assert is_valid_youtube_url(invalid_url) == False, "URL non-YouTube harus ditolak!"
assert is_valid_youtube_url(empty_url) == False, "URL kosong harus ditolak!"

# 2C. Uji penanganan error unduh URL invalid / video tidak ada
ok_dl_inv, msg_inv, title_inv = download_youtube_audio("https://youtu.be/invalid_video_id_9999999999")
assert ok_dl_inv == False, "Unduhan URL invalid harus menghasilkan False!"
print(f"Penanganan error download URL invalid: '{msg_inv}'")

# 2D. Uji unduhan audio riil YouTube (video uji 19 detik 'Me at the zoo')
print("Menguji unduhan nyata dan konversi audio YouTube ke MP3...")
ok_live, path_live, title_live = download_youtube_audio("https://www.youtube.com/watch?v=jNQXAC9IVRw")
print(f"Hasil download live: OK={ok_live}, Judul='{title_live}', Path='{path_live}'")
assert ok_live == True, f"Unduhan video valid harus berhasil! Error: {path_live}"
assert path_live and os.path.exists(path_live), "File MP3 hasil ekstraksi harus ada di disk!"
assert os.path.getsize(path_live) > 0, "Ukuran file MP3 harus lebih besar dari 0 byte!"
print(f"Ukuran MP3 terkonfirmasi: {os.path.getsize(path_live):,} byte")

# Bersihkan file uji sementara
try:
    os.remove(path_live)
    print("File uji sementara berhasil dibersihkan.")
except Exception:
    pass

print(">>> TEST 2 BERHASIL! (FFmpeg, FFprobe, validasi URL, penanganan error & konversi YouTube terverifikasi 100%)")

# -------------------------------------------------------------
# TEST 3: Ekspor Helper (TXT & JSON)
# -------------------------------------------------------------
print("\n[TEST 3] Pengujian Format Ekspor TXT & JSON...")
meta_dummy = {
    "source_type": "Teks Manual",
    "source_label": "Pengujian Unit",
    "whisper_model": "small",
    "summarizer_model": "LazarusNLP/IndoNanoT5-base",
    "audio_duration_sec": 0.0,
    "transcribe_time_sec": 0.0,
    "rtf": 0.0,
    "summary_time_sec": 1.2,
    "raw_word_count": 50,
    "clean_word_count": 45,
    "removed_filler_count": 5,
    "reduction_percentage": 10.0,
    "summary_word_count": 20
}

txt_out = create_export_txt("Raw transcript", "Cleaned transcript", "Summary", meta_dummy)
json_out = create_export_json("Raw transcript", "Cleaned transcript", "Summary", meta_dummy)

assert "LAPORAN TRANSKRIPSI" in txt_out, "Format TXT harus memuat header!"
parsed_json = json.loads(json_out)
assert parsed_json["summary"] == "Summary", "Format JSON harus memuat ringkasan!"
assert parsed_json["metadata"]["whisper_model"] == "small", "Format JSON harus memuat metadata!"
print(">>> TEST 3 BERHASIL! (Ekspor TXT dan JSON terverifikasi valid)")

# -------------------------------------------------------------
# TEST 4: Pengujian Audio Transcriber dengan file WAV sintetis
# -------------------------------------------------------------
print("\n[TEST 4] Pengujian Transcriber Audio...")
# Uji penanganan file tidak ada
_, _, _, _, ok_miss, err_miss = transcribe_audio("non_existent_file.mp3")
assert ok_miss == False, "File tidak ada harus mengembalikan False!"

# Buat file audio WAV valid berdurasi 1 detik (nada sinus 440Hz / tone pendek)
test_wav = r"d:\STT\temp_data\test_tone.wav"
os.makedirs(os.path.dirname(test_wav), exist_ok=True)

with wave.open(test_wav, 'w') as wf:
    wf.setnchannels(1)        # mono
    wf.setsampwidth(2)        # 16-bit
    wf.setframerate(16000)    # 16kHz
    # Tulis 16000 sample silence/low noise
    data = struct.pack('<' + ('h' * 16000), *([0] * 16000))
    wf.writeframes(data)

dur = get_audio_duration_seconds(test_wav)
print(f"Durasi file WAV uji: {dur:.2f} detik")
assert dur >= 0.9, f"Durasi harus mendekati 1 detik, didapat: {dur}"

# Transkripsikan audio sintetis ini dengan Whisper small
print("Menjalankan inferensi Whisper (small) pada audio uji...")
txt_tr, t_sec, dur_sec, rtf_val, ok_tr, err_tr = transcribe_audio(test_wav, model_size="small")
print(f"Hasil Transkripsi: OK={ok_tr}, Waktu={t_sec}s, Durasi={dur_sec}s, RTF={rtf_val}")
assert ok_tr == True, f"Transkripsi audio valid harus berhasil! Error: {err_tr}"
print(">>> TEST 4 BERHASIL! (Inference Whisper small berhasil dimuat dan dijalankan)")

# -------------------------------------------------------------
# TEST 5: Pengujian Summarizer IndoNanoT5 (Notebook Test Text)
# -------------------------------------------------------------
print("\n[TEST 5] Pengujian Peringkasan Hybrid IndoNanoT5...")
# Gunakan teks verifikasi asli dari Cell 17 notebook Colab
colab_test_text = (
    "kesehatan mental itu penting banget buat kehidupan kita sehari-hari. "
    "banyak orang yang masih menganggap remeh masalah psikologis padahal dampaknya "
    "sangat besar ke produktivitas dan hubungan sosial kita. "
    "caranya adalah dengan belajar mengelola stres, tidur yang cukup, dan "
    "membangun relasi yang sehat dengan orang-orang di sekitar kita. "
    "kalau kamu merasa overwhelmed jangan ragu untuk minta bantuan profesional "
    "karena itu bukan tanda kelemahan tapi justru tanda keberanian."
)

# Uji penanganan teks terlalu pendek (<30 kata)
_, _, ok_short, err_short = generate_hybrid_summary("teks ini hanya sedikit kata saja")
assert ok_short == False, "Teks terlalu pendek harus terdeteksi dan tidak dipaksakan!"
print(f"Penanganan teks pendek: '{err_short}'")

# Uji perangkuman teks verifikasi notebook
print("Menjalankan perangkuman hybrid pada teks verifikasi notebook Colab...")
sum_out, sum_meta, ok_sum, err_sum = generate_hybrid_summary(colab_test_text, target_mode="standar")

print(f"\nHasil Ringkasan:\n{sum_out}")
print(f"Metadata: {sum_meta}")
assert ok_sum == True, f"Perangkuman harus berhasil! Error: {err_sum}"
assert len(sum_out.split()) >= 30, "Ringkasan harus berisi kalimat lengkap!"
print(">>> TEST 5 BERHASIL! (Model IndoNanoT5-base berhasil dimuat dan merangkum secara hybrid)")

print("\n[TEST 6] Pengujian Optimasi Pipeline, Ekstraksi Kalimat Minim Titik, Cache & RAM Guard...")
from core.pipeline import process_audio_pipeline
from core.transcriber import unload_whisper_model, get_loaded_whisper_models
from core.summarizer import unload_summarizer_model, is_summarizer_loaded

# 6A. Uji extract_key_sentences pada teks panjang tanpa tanda baca titik
no_dot_text = (
    "kesehatan mental sangat penting bagi kehidupan sehari hari karena berdampak besar pada produktivitas, "
    "banyak orang mengabaikan gejala stres berkepanjangan hingga memengaruhi kualitas tidur, "
    "membangun kebiasaan istirahat cukup dan olahraga teratur membantu menjaga keseimbangan emosi, "
    "dukungan sosial dari keluarga dan teman terdekat membuat seseorang lebih tangguh menghadapi tekanan, "
    "jika tekanan terasa terlalu berat segera konsultasi dengan tenaga profesional untuk penanganan tepat"
)
extracted_no_dot = extract_key_sentences(no_dot_text, n=2)
assert len(extracted_no_dot.split()) >= 15, "Ekstraksi kalimat kunci pada teks minim titik harus menghasilkan ringkasan ekstraktif valid!"

# 6B. Uji pipeline penuh pada audio demo 45s (eksekusi model riil) + verifikasi cache hit
demo_audio = r"d:\STT\demo_data\contoh_podcast_45s.mp3"
assert os.path.isfile(demo_audio), "File audio demo contoh_podcast_45s.mp3 harus tersedia!"

res_live, ok_live_pipe, err_live_pipe = process_audio_pipeline(
    audio_path=demo_audio,
    whisper_model="small",
    enable_cleaning=True,
    summary_mode="standar",
    source_label="Demo Podcast 45s",
    use_cache=True,
)
assert ok_live_pipe and res_live is not None, f"Pipeline demo harus berhasil! Error: {err_live_pipe}"
assert len(res_live["raw_transcript"].split()) >= 50, "Transkrip demo harus terisi!"
assert len(res_live["summary"].split()) >= 30, "Ringkasan demo harus terisi!"

# Uji pemanggilan ulang dengan cache aktif
res_cached, ok_cached_pipe, _ = process_audio_pipeline(
    audio_path=demo_audio,
    whisper_model="small",
    enable_cleaning=True,
    summary_mode="standar",
    source_label="Demo Podcast 45s",
    use_cache=True,
)
assert ok_cached_pipe and res_cached is not None and res_cached.get("from_cache") is True, "Pemanggilan kedua harus terdeteksi dari cache!"

# 6C. Verifikasi pelepasan cache model RAM berjalan bersih
unload_whisper_model()
unload_summarizer_model()
assert get_loaded_whisper_models() == [], "Cache Whisper harus kosong setelah unload!"
assert is_summarizer_loaded() is False, "Cache IndoNanoT5 harus kosong setelah unload!"
print(">>> TEST 6 BERHASIL! (Pipeline optimasi, fallback ekstraktif, cache demo riil & unload RAM terverifikasi)")

print("\n" + "=" * 60)
print("SELURUH PENGUJIAN (TEST 1 - 6) BERHASIL 100% TANPA ERROR!")
print("=" * 60)
