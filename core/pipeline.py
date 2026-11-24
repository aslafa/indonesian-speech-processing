import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

from core.config import SUMMARIZER_MODEL_NAME
from core.memory_utils import force_gc, get_memory_stats
from core.preprocessor import clean_text_id, get_cleaning_statistics
from core.result_cache import (
    build_pipeline_cache_key,
    compute_file_sha256,
    get_cached_entry,
    save_cached_entry,
)
from core.summarizer import generate_hybrid_summary
from core.transcriber import transcribe_audio_detailed


def process_audio_pipeline(
    audio_path: str,
    whisper_model: str = "small",
    enable_cleaning: bool = True,
    summary_mode: str = "standar",
    source_label: str = "Audio",
    source_type: str = "Audio",
    use_cache: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
    """
    Menjalankan pipeline lengkap STT -> Cleaning -> Hybrid Summarization secara berurutan:
    - Mengecek cache berbasis hash audio + parameter jika use_cache=True.
    - Lazy-load Whisper terlebih dahulu, lalu IndoNanoT5 hanya setelah transkripsi selesai.
    - Melepaskan objek sementara setelah setiap tahap.

    Returns:
        (result_dict, success, error_message)
    """
    if not audio_path or not os.path.exists(audio_path):
        return None, False, f"File audio tidak ditemukan: {audio_path}"

    audio_hash = compute_file_sha256(audio_path) if use_cache else None
    cache_key = (
        build_pipeline_cache_key(audio_hash, whisper_model, enable_cleaning, summary_mode)
        if audio_hash
        else None
    )

    if use_cache and cache_key:
        cached = get_cached_entry(cache_key)
        if cached:
            if progress_callback:
                progress_callback(100, "Hasil ditemukan di cache pemrosesan sebelumnya.")
            res = dict(cached)
            res["source_label"] = source_label
            res["source_type"] = source_type
            res["from_cache"] = True
            return res, True, None

    total_start = time.time()

    # 1. Transkripsi Audio (Whisper lazy-loaded di dalam fungsi)
    if progress_callback:
        progress_callback(25, f"Menjalankan transkripsi audio dengan faster-whisper ({whisper_model})...")

    def _tr_cb(msg: str) -> None:
        if progress_callback:
            progress_callback(40, msg)

    raw_transcript, transcribe_time, audio_duration, rtf_value, ok_tr, err_tr, tr_from_cache = (
        transcribe_audio_detailed(
            file_path=audio_path,
            model_size=whisper_model,
            progress_callback=_tr_cb,
            use_cache=use_cache,
        )
    )

    if not ok_tr:
        return None, False, err_tr or "Transkripsi gagal."

    # 2. Pembersihan Filler Words Bahasa Indonesia
    if progress_callback:
        progress_callback(65, "Membersihkan filler words dan menormalisasi teks...")

    cleaned_transcript = clean_text_id(raw_transcript) if enable_cleaning else raw_transcript
    cleaning_stats = get_cleaning_statistics(raw_transcript, cleaned_transcript)

    # Lepaskan referensi sementara sebelum memuat IndoNanoT5
    force_gc()

    # 3. Peringkasan Hybrid Extractive-Abstractive (IndoNanoT5 lazy-loaded di sini)
    if progress_callback:
        progress_callback(80, "Menghasilkan ringkasan hybrid (IndoNanoT5 + Extractive)...")

    def _sum_cb(msg: str) -> None:
        if progress_callback:
            progress_callback(85, msg)

    text_to_summarize = cleaned_transcript if cleaned_transcript else raw_transcript
    sum_text, sum_meta, ok_sum, err_sum = generate_hybrid_summary(
        text=text_to_summarize,
        target_mode=summary_mode,
        progress_callback=_sum_cb,
    )

    if not ok_sum and not sum_text:
        summary_result = "[Teks terlalu pendek untuk diringkas menjadi ringkasan eksekutif]"
    else:
        summary_result = sum_text

    summary_time = float(sum_meta.get("elapsed_time_sec", 0.0))
    total_elapsed = round(time.time() - total_start, 2)
    mem_after = get_memory_stats()

    result: Dict[str, Any] = {
        "source_label": source_label,
        "source_type": source_type,
        "whisper_model": whisper_model,
        "summarizer_model": SUMMARIZER_MODEL_NAME,
        "raw_transcript": raw_transcript,
        "cleaned_transcript": cleaned_transcript,
        "summary": summary_result,
        "summary_warning": err_sum if (not ok_sum and not sum_text) else None,
        "audio_duration_sec": audio_duration,
        "transcribe_time_sec": transcribe_time,
        "rtf": rtf_value,
        "summary_time_sec": summary_time,
        "total_time_sec": total_elapsed,
        "raw_word_count": cleaning_stats["raw_word_count"],
        "clean_word_count": cleaning_stats["clean_word_count"],
        "removed_filler_count": cleaning_stats["removed_filler_count"],
        "reduction_percentage": cleaning_stats["reduction_percentage"],
        "summary_word_count": len(summary_result.split()) if summary_result else 0,
        "chunks_processed": sum_meta.get("chunks_processed", 0),
        "from_cache": False,
        "transcript_from_cache": tr_from_cache,
        "process_rss_mb": mem_after.get("process_rss_mb", 0.0),
        "system_ram_percent": mem_after.get("system_percent", 0.0),
    }

    if use_cache and cache_key and raw_transcript:
        save_cached_entry(cache_key, result)

    if progress_callback:
        progress_callback(100, "Pemrosesan selesai!")

    return result, True, None

