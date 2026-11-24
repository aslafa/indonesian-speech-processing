import os
import sys
import time
import uuid
import pandas as pd
import altair as alt
import streamlit as st

# Pastikan root workspace terdaftar di sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.config import (
    DEVICE,
    COMPUTE_TYPE,
    WHISPER_AVAILABLE_MODELS,
    DEFAULT_WHISPER_MODEL,
    SUMMARIZER_MODEL_NAME,
    TEMP_DIR,
    DEMO_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_CPU_THREADS,
    SUMMARIZER_NUM_BEAMS,
    SUMMARIZER_BATCH_SIZE,
)
from core.memory_utils import cleanup_stale_temp_files, get_memory_stats
from core.preprocessor import clean_text_id, get_cleaning_statistics
from core.transcriber import get_loaded_whisper_models, unload_whisper_model
from core.summarizer import generate_hybrid_summary, is_summarizer_loaded, unload_summarizer_model
from core.audio_downloader import (
    download_youtube_audio,
    is_valid_youtube_url,
    check_ffmpeg_availability,
)
from core.pipeline import process_audio_pipeline
from utils.export_helpers import create_export_txt, create_export_json

DEMO_AUDIO_PATH = os.path.join(DEMO_DIR, "contoh_podcast_45s.mp3")

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Indonesian Speech-to-Text",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
css_file = os.path.join(BASE_DIR, "assets", "styles.css")
if os.path.exists(css_file):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Bersihkan file temporary usang (> 1 jam) saat sesi dimulai
if "temp_cleaned_once" not in st.session_state:
    cleanup_stale_temp_files(max_age_sec=3600)
    st.session_state["temp_cleaned_once"] = True

# Inisialisasi Session State
if "processed_result" not in st.session_state:
    st.session_state["processed_result"] = None
if "uploaded_file_sig" not in st.session_state:
    st.session_state["uploaded_file_sig"] = None
if "uploaded_file_path" not in st.session_state:
    st.session_state["uploaded_file_path"] = None


def _save_uploaded_file_once(uploaded_file) -> str:
    """
    Menyimpan file unggahan ke TEMP_DIR hanya sekali per file unik (mencegah duplikasi
    file sementara dan pemborosan memori saat Streamlit melakukan rerun).
    """
    file_sig = f"{uploaded_file.name}:{uploaded_file.size}"
    existing_path = st.session_state.get("uploaded_file_path")
    if (
        st.session_state.get("uploaded_file_sig") == file_sig
        and existing_path
        and os.path.isfile(existing_path)
    ):
        return existing_path

    if existing_path and os.path.isfile(existing_path):
        try:
            os.remove(existing_path)
        except Exception:
            pass

    os.makedirs(TEMP_DIR, exist_ok=True)
    temp_ext = os.path.splitext(uploaded_file.name)[1] or ".audio"
    target_path = os.path.join(TEMP_DIR, f"upload_{uuid.uuid4().hex[:10]}{temp_ext}")
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state["uploaded_file_sig"] = file_sig
    st.session_state["uploaded_file_path"] = target_path
    return target_path


# Sidebar: Pengaturan Model dan Status Sistem
with st.sidebar:
    st.markdown('<div class="sidebar-title">Indonesian Speech-to-Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Platform Transkripsi dan Pengolahan Teks</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Pengaturan Model</div>', unsafe_allow_html=True)

    selected_whisper_model = st.selectbox(
        "Model Whisper",
        options=WHISPER_AVAILABLE_MODELS,
        index=0 if DEFAULT_WHISPER_MODEL == "small" else 1,
        help="Small dioptimalkan untuk kecepatan inferensi CPU; Medium memiliki akurasi leksikal lebih tinggi."
    )

    enable_cleaning = st.checkbox(
        "Pembersihan Kata Pengisi",
        value=True,
        help="Menghapus kata pengisi percakapan santai (ee, umm, eh, anu, ya kan, gitu loh, dll.) serta normalisasi vokal berulang."
    )

    summary_mode = st.select_slider(
        "Panjang Ringkasan",
        options=["singkat", "standar", "terperinci"],
        value="standar",
        help="Singkat: ~60-130 kata, Standar: ~80-250 kata, Terperinci: ~150-260 kata."
    )

    use_result_cache = st.checkbox(
        "Gunakan Cache Pemrosesan",
        value=True,
        help="Menggunakan hasil pemrosesan tersimpan jika berkas audio dan parameter identik."
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Status Sistem</div>', unsafe_allow_html=True)

    if DEVICE == "cuda":
        st.markdown(
            f'<div class="system-status active">Komputasi: GPU CUDA ({COMPUTE_TYPE})</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="system-status">Komputasi: CPU ({COMPUTE_TYPE}, {WHISPER_CPU_THREADS} threads)</div>',
            unsafe_allow_html=True
        )

    @st.cache_resource(show_spinner=False)
    def _cached_ffmpeg_check():
        return check_ffmpeg_availability()

    ffmpeg_ready, ffmpeg_msg, _ = _cached_ffmpeg_check()
    if ffmpeg_ready:
        st.markdown(
            '<div class="system-status active">FFmpeg: Terpasang (Aktif)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="system-status">FFmpeg: {ffmpeg_msg}</div>',
            unsafe_allow_html=True
        )

    mem_now = get_memory_stats()
    loaded_whisper = get_loaded_whisper_models()
    loaded_t5 = is_summarizer_loaded()
    st.caption(
        f"RAM Sistem: **{mem_now['system_percent']}%** (Tersedia: {mem_now['system_available_mb']:.0f} MB)\n\n"
        f"Memori Proses: **{mem_now['process_rss_mb']:.0f} MB**\n\n"
        f"Model Aktif: Whisper={loaded_whisper[0] if loaded_whisper else 'Belum Dimuat'}, T5={'Aktif' if loaded_t5 else 'Belum Dimuat'}"
    )

    if st.button("Bebaskan Memori Model", width="stretch"):
        w_cnt = unload_whisper_model()
        t_freed = unload_summarizer_model()
        if w_cnt > 0 or t_freed:
            st.success("Memori model berhasil dilepaskan.")
        else:
            st.info("Tidak ada model aktif dalam memori.")
        st.rerun()

    st.markdown("---")
    st.caption("Engine: Faster-Whisper INT8 & LazarusNLP/IndoNanoT5-base.")


# Area Header Utama
st.markdown("""
<div class="app-header">
    <div class="app-badge">Platform Speech-to-Text</div>
    <div class="app-title">Indonesian Speech-to-Text</div>
    <div class="app-subtitle">
        Sistem transkripsi wicara berbasis Faster-Whisper dan peringkasan teks hybrid IndoNanoT5-base untuk percakapan dan narasi Bahasa Indonesia.
    </div>
</div>
""", unsafe_allow_html=True)

# Struktur Navigasi Halaman
tab_processing, tab_eval = st.tabs(["Pemrosesan Transkripsi", "Evaluasi Model"])

# ==============================================================================
# TAB 1: PEMROSESAN TRANSKRIPSI
# ==============================================================================
with tab_processing:
    input_options = [
        "Unggah Audio",
        "URL YouTube",
        "Teks Langsung"
    ]
    selected_input = st.radio(
        "Metode Masukan:",
        input_options,
        index=0,
        horizontal=True
    )

    audio_path_to_process = None
    is_temp_audio = False
    direct_text_to_process = None
    input_source_label = ""
    yt_url = ""

    if selected_input == "Unggah Audio":
        uploaded_file = st.file_uploader(
            "Pilih berkas audio (format yang didukung: MP3, WAV, M4A, OGG, FLAC, WEBM):",
            type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
            help="Mendukung format audio standar untuk decoding langsung."
        )

        use_demo_sample = False
        if os.path.isfile(DEMO_AUDIO_PATH):
            with st.expander("Gunakan Berkas Sampel Pengujian (Audio 45 Detik)"):
                st.write("Berkas narasi percakapan Bahasa Indonesia berdurasi 45 detik untuk demonstrasi dan evaluasi cepat sistem.")
                st.audio(DEMO_AUDIO_PATH)
                if st.button("Muat Audio Sampel Sebagai Masukan"):
                    use_demo_sample = True

        if use_demo_sample:
            audio_path_to_process = DEMO_AUDIO_PATH
            is_temp_audio = False
            input_source_label = "Sampel Evaluasi (45 Detik)"
            st.info("Audio sampel 45 detik siap diproses.")
        elif uploaded_file is not None:
            st.audio(uploaded_file, format=uploaded_file.type)
            audio_path_to_process = _save_uploaded_file_once(uploaded_file)
            is_temp_audio = True
            input_source_label = f"Berkas Audio: {uploaded_file.name}"

    elif selected_input == "URL YouTube":
        yt_url = st.text_input(
            "Tautan Video YouTube:",
            placeholder="https://www.youtube.com/watch?v=... atau https://youtu.be/..."
        )
        if yt_url:
            if not is_valid_youtube_url(yt_url):
                st.warning("Format URL YouTube tidak valid. Harap periksa kembali tautan Anda.")
            else:
                input_source_label = f"YouTube: {yt_url}"

    elif selected_input == "Teks Langsung":
        direct_text = st.text_area(
            "Teks Masukan:",
            height=200,
            placeholder="Masukkan atau tempel teks bahasa Indonesia untuk diproses langsung..."
        )
        if direct_text.strip():
            direct_text_to_process = direct_text.strip()
            input_source_label = "Teks Manual"

    # Tombol Eksekusi Pemrosesan
    st.markdown("<br>", unsafe_allow_html=True)
    process_button = st.button("Mulai Pemrosesan", type="primary", width="stretch")

    if process_button:
        can_proceed = False
        if selected_input == "Unggah Audio" and audio_path_to_process:
            can_proceed = True
        elif selected_input == "URL YouTube" and yt_url and is_valid_youtube_url(yt_url):
            can_proceed = True
        elif selected_input == "Teks Langsung" and direct_text_to_process:
            can_proceed = True

        if not can_proceed:
            st.error("Tentukan masukan audio atau teks yang valid terlebih dahulu.")
        else:
            progress_bar = st.progress(0, text="Memulai pemrosesan...")
            status_box = st.empty()

            try:
                # Unduh audio dari YouTube jika dipilih
                if selected_input == "URL YouTube":
                    progress_bar.progress(10, text="Mengunduh audio dari YouTube...")
                    status_box.info("Mengunduh stream audio dari YouTube...")
                    ok_dl, audio_res, video_title = download_youtube_audio(
                        yt_url,
                        convert_to_mp3=False,
                        use_cache=use_result_cache,
                    )
                    if not ok_dl:
                        progress_bar.empty()
                        status_box.empty()
                        st.error(f"Gagal Mengambil Audio YouTube:\n\n{audio_res}")
                        if "403" in audio_res or "Forbidden" in audio_res or "IP server" in audio_res:
                            st.info(
                                "💡 **Saran:** Silakan buka video di YouTube pada perangkat Anda, "
                                "simpan audionya (MP3/M4A/WAV), lalu beralih ke tab pilihan **'Unggah Audio'** di atas."
                            )
                        audio_path_to_process = None
                    else:
                        audio_path_to_process = audio_res
                        is_temp_audio = not use_result_cache
                        if video_title:
                            input_source_label = f"YouTube: {video_title}"

                if audio_path_to_process:
                    def _pipeline_progress(pct: int, msg: str) -> None:
                        progress_bar.progress(min(max(pct, 0), 100), text=msg)
                        status_box.info(msg)

                    res, ok_pipe, err_pipe = process_audio_pipeline(
                        audio_path=audio_path_to_process,
                        whisper_model=selected_whisper_model,
                        enable_cleaning=enable_cleaning,
                        summary_mode=summary_mode,
                        source_label=input_source_label,
                        source_type=selected_input,
                        use_cache=use_result_cache,
                        progress_callback=_pipeline_progress,
                    )
                    if not ok_pipe or res is None:
                        st.error(f"Pemrosesan gagal: {err_pipe}")
                        st.stop()

                    if res.get("summary_warning"):
                        st.warning(f"Catatan peringkasan: {res['summary_warning']}")

                    st.session_state["processed_result"] = res
                else:
                    # Masukan teks langsung
                    progress_bar.progress(45, text="Membersihkan kata pengisi...")
                    status_box.info("Melakukan pembersihan filler words dan normalisasi teks...")
                    raw_transcript = direct_text_to_process
                    cleaned_transcript = clean_text_id(raw_transcript) if enable_cleaning else raw_transcript
                    cleaning_stats = get_cleaning_statistics(raw_transcript, cleaned_transcript)

                    progress_bar.progress(75, text="Menghasilkan ringkasan hybrid...")
                    status_box.info("Merangkum teks dengan model IndoNanoT5-base...")
                    text_to_summarize = cleaned_transcript if cleaned_transcript else raw_transcript
                    sum_text, sum_meta, ok_sum, err_sum = generate_hybrid_summary(
                        text=text_to_summarize,
                        target_mode=summary_mode,
                        progress_callback=lambda m: status_box.info(m),
                    )
                    if not ok_sum and not sum_text:
                        st.warning(f"Catatan peringkasan: {err_sum}")
                        summary_result = "[Teks terlalu pendek untuk diringkas menjadi ringkasan eksekutif]"
                    else:
                        summary_result = sum_text

                    mem_after = get_memory_stats()
                    st.session_state["processed_result"] = {
                        "source_label": input_source_label,
                        "source_type": selected_input,
                        "whisper_model": "N/A (Masukan Teks)",
                        "summarizer_model": SUMMARIZER_MODEL_NAME,
                        "raw_transcript": raw_transcript,
                        "cleaned_transcript": cleaned_transcript,
                        "summary": summary_result,
                        "audio_duration_sec": 0.0,
                        "transcribe_time_sec": 0.0,
                        "rtf": 0.0,
                        "summary_time_sec": sum_meta.get("elapsed_time_sec", 0.0),
                        "total_time_sec": sum_meta.get("elapsed_time_sec", 0.0),
                        "raw_word_count": cleaning_stats["raw_word_count"],
                        "clean_word_count": cleaning_stats["clean_word_count"],
                        "removed_filler_count": cleaning_stats["removed_filler_count"],
                        "reduction_percentage": cleaning_stats["reduction_percentage"],
                        "summary_word_count": len(summary_result.split()) if summary_result else 0,
                        "from_cache": False,
                        "process_rss_mb": mem_after.get("process_rss_mb", 0.0),
                        "system_ram_percent": mem_after.get("system_percent", 0.0),
                    }

                progress_bar.progress(100, text="Pemrosesan selesai.")
                status_box.success("Pemrosesan berhasil diselesaikan.")
                time.sleep(0.4)
                status_box.empty()
                progress_bar.empty()

            except Exception as ex:
                st.error(f"Terjadi kesalahan saat pemrosesan: {ex}")
            finally:
                if is_temp_audio and audio_path_to_process and os.path.exists(audio_path_to_process):
                    try:
                        os.remove(audio_path_to_process)
                    except Exception:
                        pass
                if st.session_state.get("uploaded_file_path") == audio_path_to_process and is_temp_audio:
                    st.session_state["uploaded_file_path"] = None
                    st.session_state["uploaded_file_sig"] = None

    # Halaman Hasil Pemrosesan
    # Urutan terstruktur:
    # 1. Informasi sumber audio.
    # 2. Model yang digunakan.
    # 3. Durasi audio dan waktu transkripsi.
    # 4. RTF dan waktu peringkasan.
    # 5. Ringkasan Eksekutif.
    # 6. Tab Transkrip Bersih dan Transkrip Mentah.
    # 7. Tombol unduh TXT dan JSON.
    if st.session_state["processed_result"]:
        res = st.session_state["processed_result"]
        st.markdown("---")
        st.subheader("Hasil Pemrosesan")

        if res.get("from_cache"):
            cached_at = res.get("cached_at", "")
            time_str = f" ({cached_at[:19]})" if cached_at else ""
            st.markdown(
                f'<div class="notice-box"><strong>Status Cache:</strong> Hasil diambil dari cache pemrosesan sebelumnya{time_str}.</div>',
                unsafe_allow_html=True
            )

        # 1. Informasi sumber audio & 2. Model yang digunakan
        c_meta1, c_meta2 = st.columns(2)
        with c_meta1:
            st.markdown(f"**Sumber Audio:** {res['source_label']}")
        with c_meta2:
            st.markdown(f"**Model Transkripsi:** Faster-Whisper ({res.get('whisper_model', '-')})")

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Durasi audio dan waktu transkripsi & 4. RTF dan waktu peringkasan
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dur_txt = f"{res['audio_duration_sec']:.1f} s" if res["audio_duration_sec"] > 0 else f"{res['raw_word_count']} kata"
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Durasi Audio</div>'
                f'<div class="metric-value">{dur_txt}</div>'
                f'<div class="metric-sub">Panjang Masukan</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col2:
            tr_txt = f"{res['transcribe_time_sec']:.2f} s" if res["transcribe_time_sec"] > 0 else "Dilewati"
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Waktu Transkripsi</div>'
                f'<div class="metric-value">{tr_txt}</div>'
                f'<div class="metric-sub">Faster-Whisper</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col3:
            rtf_txt = f"{res['rtf']:.3f}" if res["rtf"] > 0 else "-"
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Real-Time Factor (RTF)</div>'
                f'<div class="metric-value">{rtf_txt}</div>'
                f'<div class="metric-sub">Rasio Waktu/Durasi</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-label">Waktu Peringkasan</div>'
                f'<div class="metric-value">{res["summary_time_sec"]:.2f} s</div>'
                f'<div class="metric-sub">IndoNanoT5-base</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # 5. Ringkasan Eksekutif
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Ringkasan Eksekutif**")
        st.markdown(
            f'<div class="summary-container">{res["summary"]}</div>',
            unsafe_allow_html=True
        )

        # 6. Tab Transkrip Bersih dan Transkrip Mentah
        tab_clean, tab_raw = st.tabs(["Transkrip Bersih", "Transkrip Mentah"])
        with tab_clean:
            st.text_area(
                "Transkrip Bersih (Tanpa Kata Pengisi):",
                value=res["cleaned_transcript"],
                height=220,
                label_visibility="collapsed"
            )
            st.caption(f"Jumlah kata bersih: {res['clean_word_count']} kata | Kata pengisi dihapus: {res['removed_filler_count']} kata ({res['reduction_percentage']}%)")

        with tab_raw:
            st.text_area(
                "Transkrip Mentah (Keluaran Asli Model):",
                value=res["raw_transcript"],
                height=220,
                label_visibility="collapsed"
            )
            st.caption(f"Jumlah kata asli: {res['raw_word_count']} kata")

        # 7. Tombol unduh TXT dan JSON
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Unduh Hasil**")
        d_col1, d_col2 = st.columns(2)

        txt_content = create_export_txt(
            raw_transcript=res["raw_transcript"],
            cleaned_transcript=res["cleaned_transcript"],
            summary=res["summary"],
            metadata=res
        )
        json_content = create_export_json(
            raw_transcript=res["raw_transcript"],
            cleaned_transcript=res["cleaned_transcript"],
            summary=res["summary"],
            metadata=res
        )

        with d_col1:
            st.download_button(
                label="Unduh TXT",
                data=txt_content,
                file_name="laporan_transkripsi_ringkasan.txt",
                mime="text/plain",
                width="stretch"
            )
        with d_col2:
            st.download_button(
                label="Unduh JSON",
                data=json_content,
                file_name="hasil_transkripsi_ringkasan.json",
                mime="application/json",
                width="stretch"
            )


# ==============================================================================
# TAB 2: EVALUASI MODEL
# ==============================================================================
with tab_eval:
    st.subheader("Evaluasi Model")
    st.markdown("""
    Laporan kinerja komparatif arsitektur Faster-Whisper dan IndoNanoT5-base pada transkripsi dan peringkasan wicara Bahasa Indonesia.
    """)

    # 1. Perbandingan Whisper Small dan Medium
    st.markdown("### 1. Perbandingan Karakteristik Whisper Small dan Medium")
    st.markdown("""
    <table class="academic-table">
        <thead>
            <tr>
                <th>Parameter / Karakteristik</th>
                <th>Whisper Small</th>
                <th>Whisper Medium</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Jumlah Parameter Arsitektur</strong></td>
                <td class="mono">~244 Juta</td>
                <td class="mono">~769 Juta</td>
            </tr>
            <tr>
                <td><strong>Ukuran Model Kuantisasi INT8</strong></td>
                <td class="mono">~484 MB</td>
                <td class="mono">~1.53 GB</td>
            </tr>
            <tr>
                <td><strong>Alokasi Memori RAM Aktif</strong></td>
                <td class="mono">~450 – 650 MB</td>
                <td class="mono">~950 – 1,300 MB</td>
            </tr>
            <tr>
                <td><strong>Real-Time Factor (RTF pada CPU)</strong></td>
                <td class="mono"><strong>0.19 – 0.45</strong> (2x-5x lebih cepat dari waktu nyata)</td>
                <td class="mono"><strong>0.84 – 0.88</strong> (Sedikit di bawah waktu nyata)</td>
            </tr>
            <tr>
                <td><strong>Ketepatan Leksikal & Konteks</strong></td>
                <td>Sangat baik untuk narasi monolog, podcast terstruktur, dan ujaran jelas</td>
                <td>Akurasi leksikal superior untuk multi-pembicara, tumpang tindih, dan jargon teknis</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Metrik Ground Truth Penelitian (WER, CER, ROUGE)
    st.markdown("### 2. Metrik Akurasi Ground Truth (Dataset 40 Episode Podcast Satu Persen)")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">Word Error Rate (WER)</div>'
            '<div class="metric-value">~16.5%</div>'
            '<div class="metric-sub">Kategori: Baik</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">Character Error Rate (CER)</div>'
            '<div class="metric-value">~6.8%</div>'
            '<div class="metric-sub">Kategori: Sangat Baik</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">ROUGE-1 F1</div>'
            '<div class="metric-value">0.528</div>'
            '<div class="metric-sub">Presisi Topik Utama</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">ROUGE-2 F1</div>'
            '<div class="metric-value">0.466</div>'
            '<div class="metric-sub">Koherensi Frasa</div>'
            '</div>',
            unsafe_allow_html=True
        )
    with m_col5:
        st.markdown(
            '<div class="metric-box">'
            '<div class="metric-label">ROUGE-L F1</div>'
            '<div class="metric-value">0.469</div>'
            '<div class="metric-sub">Struktur Kalimat</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # 3. Grafik Sederhana: Perbandingan RTF dan Percepatan Waktu
    st.markdown("### 3. Grafik Analisis Kinerja Komparatif")
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("**Perbandingan Real-Time Factor (RTF) pada CPU**")
        df_rtf = pd.DataFrame({
            "Durasi": ["45 Detik", "4 Menit", "12 Menit", "25.6 Menit"],
            "Whisper Small": [0.196, 0.327, 0.449, 0.420],
            "Whisper Medium": [0.842, 0.876, 0.880, 0.885],
        })
        df_rtf_melted = df_rtf.melt("Durasi", var_name="Model", value_name="RTF")

        chart_rtf = alt.Chart(df_rtf_melted).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Durasi:N", title="Durasi Pengujian Audio", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("RTF:Q", title="Real-Time Factor (RTF)", scale=alt.Scale(domain=[0, 1.2])),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=["Whisper Small", "Whisper Medium"], range=["#0F766E", "#64748B"]),
                legend=alt.Legend(title="Model", orient="top")
            ),
            xOffset="Model:N"
        ).properties(height=280)

        rule_rtf = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(
            color="#DC2626", strokeDash=[4, 4], strokeWidth=1.5
        ).encode(y="y:Q")

        st.altair_chart((chart_rtf + rule_rtf).configure_view(strokeWidth=0), width="stretch")
        st.caption("Catatan: Nilai RTF < 1.0 (garis putus merah) menandakan pemrosesan lebih cepat dari durasi audio nyata.")

    with g_col2:
        st.markdown("**Faktor Percepatan Pemrosesan Sebelum vs Sesudah Optimasi**")
        df_speedup = pd.DataFrame({
            "Skenario": [
                "45s (Small)",
                "45s (Medium)",
                "4m (Small)",
                "4m (Medium)",
                "12m (Small)",
                "25.6m (Small)"
            ],
            "Percepatan": [2.18, 3.85, 2.93, 2.58, 1.96, 2.33]
        })

        chart_speedup = alt.Chart(df_speedup).mark_bar(
            color="#0F766E", cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("Skenario:N", title="Skenario Pengujian", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Percepatan:Q", title="Faktor Percepatan (x lipat)", scale=alt.Scale(domain=[0, 4.5]))
        ).properties(height=280)

        text_speedup = chart_speedup.mark_text(
            align="center",
            baseline="bottom",
            dy=-4,
            fontSize=11,
            fontWeight=600,
            color="#0F172A"
        ).encode(
            text=alt.Text("Percepatan:Q", format=".2f")
        )

        st.altair_chart((chart_speedup + text_speedup).configure_view(strokeWidth=0), width="stretch")
        st.caption("Catatan: Rasio perbandingan waktu eksekusi baseline tanpa optimasi terhadap arsitektur teroptimasi INT8.")

    st.markdown("---")

    # 4. Hasil Pengujian Benchmark Empiris (RTF dan Waktu Pemrosesan)
    st.markdown("### 4. Hasil Pengujian Benchmark Empiris Sebelum dan Sesudah Optimasi")
    st.markdown("""
    Pengujian empiris dilakukan pada perangkat uji: **Intel Core i7-1355U, RAM 16 GB, OS Windows 11**:
    """)

    st.markdown("""
    <table class="academic-table">
        <thead>
            <tr>
                <th>Durasi Audio</th>
                <th>Varian Model</th>
                <th>Waktu Transkripsi (Sebelum)</th>
                <th>Waktu Transkripsi (Sesudah)</th>
                <th>RTF Sebelum</th>
                <th>RTF Sesudah</th>
                <th>Waktu Peringkasan (Sebelum)</th>
                <th>Waktu Peringkasan (Sesudah)</th>
                <th>Faktor Percepatan</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>45 Detik</strong></td>
                <td>Small</td>
                <td class="mono">19.31 s</td>
                <td class="mono"><strong>8.84 s</strong></td>
                <td class="mono">0.429</td>
                <td class="mono"><strong>0.196</strong></td>
                <td class="mono">33.52 s</td>
                <td class="mono"><strong>13.79 s</strong></td>
                <td class="mono"><strong>2.18x</strong></td>
            </tr>
            <tr>
                <td><strong>45 Detik</strong></td>
                <td>Medium</td>
                <td class="mono">102.05 s</td>
                <td class="mono"><strong>37.87 s</strong></td>
                <td class="mono">2.268</td>
                <td class="mono"><strong>0.842</strong></td>
                <td class="mono">97.56 s</td>
                <td class="mono"><strong>14.19 s</strong></td>
                <td class="mono"><strong>3.85x</strong></td>
            </tr>
            <tr>
                <td><strong>4 Menit</strong></td>
                <td>Small</td>
                <td class="mono">191.68 s</td>
                <td class="mono"><strong>78.44 s</strong></td>
                <td class="mono">0.799</td>
                <td class="mono"><strong>0.327</strong></td>
                <td class="mono">113.48 s</td>
                <td class="mono"><strong>25.02 s</strong></td>
                <td class="mono"><strong>2.93x</strong></td>
            </tr>
            <tr>
                <td><strong>4 Menit</strong></td>
                <td>Medium</td>
                <td class="mono">~450.0 s*</td>
                <td class="mono"><strong>210.19 s</strong></td>
                <td class="mono">~1.875*</td>
                <td class="mono"><strong>0.876</strong></td>
                <td class="mono">~180.0 s*</td>
                <td class="mono"><strong>24.37 s</strong></td>
                <td class="mono"><strong>~2.58x</strong></td>
            </tr>
            <tr>
                <td><strong>12 Menit</strong></td>
                <td>Small</td>
                <td class="mono">559.84 s</td>
                <td class="mono"><strong>323.00 s</strong></td>
                <td class="mono">0.778</td>
                <td class="mono"><strong>0.449</strong></td>
                <td class="mono">274.05 s</td>
                <td class="mono"><strong>95.99 s</strong></td>
                <td class="mono"><strong>1.96x</strong></td>
            </tr>
            <tr>
                <td><strong>25.6 Menit</strong></td>
                <td>Small</td>
                <td class="mono">~1,200 s*</td>
                <td class="mono"><strong>644.44 s</strong></td>
                <td class="mono">~0.780*</td>
                <td class="mono"><strong>0.420</strong></td>
                <td class="mono">~550 s*</td>
                <td class="mono"><strong>99.73 s</strong></td>
                <td class="mono"><strong>~2.33x</strong></td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)
    st.caption("* Catatan: Nilai bertanda bintang (*) merupakan estimasi konservatif sebelum optimasi pipeline.")

    st.markdown("---")

    # 5. Penggunaan RAM dan Efisiensi Komputasi
    st.markdown("### 5. Penggunaan RAM dan Efisiensi Memori")
    ram_col1, ram_col2 = st.columns(2)

    with ram_col1:
        st.markdown("""
        <div class="eval-card">
            <div class="eval-card-title">Alokasi Memori Model Empiris</div>
            <p class="eval-card-text">
                • <strong>Whisper Small (INT8):</strong> Alokasi memori berkisar antara <strong>450–650 MB</strong>, sangat aman untuk CPU multithread 16 GB.<br>
                • <strong>Whisper Medium (INT8):</strong> Alokasi memori berkisar antara <strong>950–1,300 MB</strong>, membutuhkan kapasitas headroom RAM lebih besar.<br>
                • <strong>IndoNanoT5-base:</strong> Alokasi memori berkisar <strong>~800 MB</strong> saat proses inferensi token decoding aktif.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with ram_col2:
        st.markdown("""
        <div class="eval-card">
            <div class="eval-card-title">Manajemen dan Proteksi Memori</div>
            <p class="eval-card-text">
                • <strong>Lazy Loading:</strong> Model Faster-Whisper dan IndoNanoT5 hanya dialokasikan ke RAM saat tahap pemrosesan dimulai.<br>
                • <strong>Pemisahan Beban:</strong> Objek buffer audio dibersihkan secara eksplisit sebelum model perangkum dialokasikan.<br>
                • <strong>Ambang Batas Proteksi:</strong> Jika sisa RAM sistem berada di bawah <strong>2,000 MB</strong> atau beban melampaui <strong>86.0%</strong>, model pasif dilepaskan dari memori.
            </p>
        </div>
        """, unsafe_allow_html=True)
