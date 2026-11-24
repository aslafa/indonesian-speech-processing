import gc
import os
import time
from typing import Tuple, Optional, Callable, Dict, Any

from core.config import (
    DEVICE,
    COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_NUM_WORKERS,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_LANGUAGE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_WORD_TIMESTAMPS,
    WHISPER_VAD_FILTER,
    WHISPER_VAD_PARAMETERS,
)
from core.memory_utils import INFERENCE_LOCK, force_gc, get_memory_stats
from core.result_cache import (
    compute_file_sha256,
    build_transcript_cache_key,
    get_cached_entry,
    save_cached_entry,
)

# Cache untuk model Whisper yang sedang aktif di memori (maksimal 1 varian untuk hemat RAM)
_WHISPER_MODELS_CACHE: Dict[str, Any] = {}


def get_audio_duration_seconds(file_path: str) -> float:
    """
    Menghitung durasi audio dalam detik menggunakan av (PyAV) atau ffprobe.
    """
    container = None
    try:
        import av
        container = av.open(file_path)
        duration_sec = float(container.duration) / av.time_base if container.duration else 0.0
        return duration_sec
    except Exception:
        pass
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
            del container

    # Fallback coba ffprobe via subprocess jika av gagal
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_loaded_whisper_models() -> list:
    """Mengembalikan daftar varian model Whisper yang saat ini berada di cache RAM."""
    return list(_WHISPER_MODELS_CACHE.keys())


def unload_whisper_model(keep_size: Optional[str] = None) -> int:
    """
    Melepaskan model Whisper dari cache RAM.
    Jika keep_size diberikan, hanya melepaskan varian selain keep_size.
    """
    global _WHISPER_MODELS_CACHE
    removed = 0
    for key in list(_WHISPER_MODELS_CACHE.keys()):
        if keep_size is not None and key == keep_size:
            continue
        model_obj = _WHISPER_MODELS_CACHE.pop(key, None)
        del model_obj
        removed += 1
    if removed > 0:
        force_gc()
    return removed


def load_whisper_model(model_size: str = "small"):
    """
    Lazy-load model Whisper ke memori dengan caching terkelola:
    - Tidak memuat ulang jika varian yang sama sudah ada di cache.
    - Otomatis melepas varian Whisper lain yang tidak aktif agar tidak menumpuk di RAM.
    - Menggunakan local_files_only terlebih dahulu untuk mempercepat pemuatan offline/cache.
    """
    global _WHISPER_MODELS_CACHE
    if model_size in _WHISPER_MODELS_CACHE:
        return _WHISPER_MODELS_CACHE[model_size]

    # Lepaskan varian Whisper lain yang sedang tidak dipakai sebelum memuat varian baru
    unload_whisper_model(keep_size=model_size)

    # Perlindungan RAM: jika RAM sistem kritis sebelum memuat Whisper, lepas cache IndoNanoT5 sementara
    mem_stats = get_memory_stats()
    if mem_stats.get("is_critical"):
        try:
            from core.summarizer import unload_summarizer_model
            unload_summarizer_model()
        except Exception:
            pass

    from faster_whisper import WhisperModel

    init_kwargs: Dict[str, Any] = {
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "num_workers": WHISPER_NUM_WORKERS,
        "download_root": None,
    }
    if DEVICE == "cpu":
        init_kwargs["cpu_threads"] = WHISPER_CPU_THREADS

    try:
        model = WhisperModel(model_size, local_files_only=True, **init_kwargs)
    except Exception:
        model = WhisperModel(model_size, local_files_only=False, **init_kwargs)

    _WHISPER_MODELS_CACHE[model_size] = model
    return model


def transcribe_audio(
    file_path: str,
    model_size: str = "small",
    progress_callback: Optional[Callable[[str], None]] = None,
    use_cache: bool = False,
) -> Tuple[str, float, float, float, bool, Optional[str]]:
    """
    Melakukan transkripsi audio menggunakan faster-whisper dengan konfigurasi teroptimasi:
    - beam_size = 1, best_of = 1 (cepat di CPU dengan kualitas tetap akurat)
    - language = 'id'
    - vad_filter = True (min_silence_duration_ms = 500, speech_pad_ms = 250)
    - condition_on_previous_text = True
    - word_timestamps = False
    - cpu_threads = WHISPER_CPU_THREADS (6 pada i7-1355U), num_workers = 1

    Returns:
        (text, elapsed_time_sec, audio_duration_sec, rtf, success, error_message)
    """
    text, elapsed, dur, rtf, ok, err, _ = transcribe_audio_detailed(
        file_path=file_path,
        model_size=model_size,
        progress_callback=progress_callback,
        use_cache=use_cache,
    )
    return text, elapsed, dur, rtf, ok, err


def transcribe_audio_detailed(
    file_path: str,
    model_size: str = "small",
    progress_callback: Optional[Callable[[str], None]] = None,
    use_cache: bool = True,
) -> Tuple[str, float, float, float, bool, Optional[str], bool]:
    """
    Versi detail dari transcribe_audio yang turut mengembalikan flag from_cache.
    Returns:
        (text, elapsed_time_sec, audio_duration_sec, rtf, success, error_message, from_cache)
    """
    if not os.path.exists(file_path):
        return "", 0.0, 0.0, 0.0, False, f"File audio tidak ditemukan di: {file_path}", False

    file_size_kb = os.path.getsize(file_path) / 1024
    if file_size_kb < 10:
        return "", 0.0, 0.0, 0.0, False, "File audio terlalu kecil atau kosong.", False

    audio_hash = compute_file_sha256(file_path) if use_cache else None
    cache_key = build_transcript_cache_key(audio_hash, model_size) if audio_hash else None
    if use_cache and cache_key:
        cached = get_cached_entry(cache_key)
        if cached:
            if progress_callback:
                progress_callback("Mengambil hasil transkripsi dari cache...")
            return (
                cached.get("raw_transcript", ""),
                float(cached.get("transcribe_time_sec", 0.0)),
                float(cached.get("audio_duration_sec", 0.0)),
                float(cached.get("rtf", 0.0)),
                True,
                None,
                True,
            )

    audio_dur = get_audio_duration_seconds(file_path)

    if progress_callback:
        progress_callback(f"Memuat model Whisper ({model_size}) pada perangkat {DEVICE.upper()}...")

    segments = None
    info = None
    start_time = time.time()

    with INFERENCE_LOCK:
        try:
            model = load_whisper_model(model_size)
        except Exception as e:
            return "", 0.0, 0.0, 0.0, False, f"Gagal memuat model Whisper: {e}", False

        if progress_callback:
            progress_callback(f"Mentranskripsikan audio ({audio_dur:.1f} detik)...")

        start_time = time.time()
        try:
            transcribe_kwargs: Dict[str, Any] = {
                "beam_size": WHISPER_BEAM_SIZE,
                "language": WHISPER_LANGUAGE,
                "vad_filter": WHISPER_VAD_FILTER,
                "condition_on_previous_text": WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                "word_timestamps": WHISPER_WORD_TIMESTAMPS,
            }
            if WHISPER_BEAM_SIZE == 1:
                transcribe_kwargs["best_of"] = WHISPER_BEST_OF
            if WHISPER_VAD_FILTER:
                transcribe_kwargs["vad_parameters"] = dict(WHISPER_VAD_PARAMETERS)

            segments, info = model.transcribe(file_path, **transcribe_kwargs)

            segment_texts = []
            for segment in segments:
                txt_part = segment.text.strip()
                if txt_part:
                    segment_texts.append(txt_part)

            full_text = " ".join(segment_texts).strip()
            elapsed = round(time.time() - start_time, 2)

            if audio_dur <= 0.0 and info is not None and getattr(info, "duration", None):
                audio_dur = float(info.duration)

            audio_dur_rounded = round(audio_dur, 2)
            rtf = round(elapsed / audio_dur, 3) if audio_dur > 0 else 0.0

            if use_cache and cache_key and full_text:
                save_cached_entry(
                    cache_key,
                    {
                        "raw_transcript": full_text,
                        "transcribe_time_sec": elapsed,
                        "audio_duration_sec": audio_dur_rounded,
                        "rtf": rtf,
                        "whisper_model": model_size,
                    },
                )

            return full_text, elapsed, audio_dur_rounded, rtf, True, None, False
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            return "", elapsed, round(audio_dur, 2), 0.0, False, f"Error saat transkripsi: {e}", False
        finally:
            # Lepaskan generator segmen & objek audio sementara segera setelah transkripsi
            if segments is not None:
                del segments
            if info is not None:
                del info
            force_gc()
