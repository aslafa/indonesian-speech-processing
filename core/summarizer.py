import re
import time
import torch
from typing import Tuple, Optional, Callable, Dict, Any, List

from core.config import (
    DEVICE,
    SUMMARIZER_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUMMARY_MIN_WORDS,
    SUMMARY_MAX_WORDS,
    SUMMARIZER_NUM_BEAMS,
    SUMMARIZER_BATCH_SIZE,
    SUMMARIZER_REPETITION_PENALTY,
    SUMMARIZER_NO_REPEAT_NGRAM_SIZE,
    TORCH_CPU_THREADS,
)
from core.memory_utils import INFERENCE_LOCK, get_memory_stats, force_gc
from core.preprocessor import extract_key_sentences

# Global cache model & tokenizer summarizer (lazy-loaded)
_SUMMARIZER_MODEL = None
_SUMMARIZER_TOKENIZER = None


def is_summarizer_loaded() -> bool:
    """Memeriksa apakah model IndoNanoT5 sudah dimuat di cache RAM."""
    return _SUMMARIZER_MODEL is not None and _SUMMARIZER_TOKENIZER is not None


def unload_summarizer_model() -> bool:
    """Melepaskan model dan tokenizer IndoNanoT5 dari cache RAM."""
    global _SUMMARIZER_MODEL, _SUMMARIZER_TOKENIZER
    was_loaded = _SUMMARIZER_MODEL is not None or _SUMMARIZER_TOKENIZER is not None
    _SUMMARIZER_MODEL = None
    _SUMMARIZER_TOKENIZER = None
    if was_loaded:
        force_gc()
    return was_loaded


def load_summarizer_model():
    """
    Lazy-load tokenizer dan model IndoNanoT5-base ke memori secara singleton/cache.
    Jika RAM sistem dalam kondisi kritis saat model belum dimuat, model Whisper yang tidak
    sedang dipakai akan dilepaskan sementara untuk mencegah kelebihan memori.
    """
    global _SUMMARIZER_MODEL, _SUMMARIZER_TOKENIZER
    if _SUMMARIZER_MODEL is not None and _SUMMARIZER_TOKENIZER is not None:
        return _SUMMARIZER_TOKENIZER, _SUMMARIZER_MODEL

    if DEVICE == "cpu":
        try:
            torch.set_num_threads(TORCH_CPU_THREADS)
        except Exception:
            pass

    # Perlindungan RAM darurat: lepaskan cache Whisper sementara hanya jika RAM sistem kritis
    mem_stats = get_memory_stats()
    if mem_stats.get("is_critical"):
        try:
            from core.transcriber import unload_whisper_model
            unload_whisper_model()
        except Exception:
            pass

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    try:
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

    model = model.to(DEVICE)
    model.eval()

    _SUMMARIZER_TOKENIZER = tokenizer
    _SUMMARIZER_MODEL = model
    return tokenizer, model


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Membagi teks panjang menjadi chunk kata dengan overlap.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def _build_generate_kwargs(
    max_new_tokens: int,
    min_new_tokens: int,
    length_penalty: float,
    num_beams: int = SUMMARIZER_NUM_BEAMS,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "num_beams": num_beams,
        "no_repeat_ngram_size": SUMMARIZER_NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": SUMMARIZER_REPETITION_PENALTY,
    }
    if num_beams > 1:
        kwargs["length_penalty"] = length_penalty
        kwargs["early_stopping"] = True
    return kwargs


def summarize_single_chunk(
    text: str,
    tokenizer,
    model,
    max_new_tokens: int = 120,
    min_new_tokens: int = 35,
    length_penalty: float = 1.5,
) -> str:
    """
    Menjalankan inferensi abstraktif model T5 pada satu potongan teks dengan torch.inference_mode().
    """
    if not text.strip():
        return ""
    inputs = None
    ids = None
    try:
        inputs = tokenizer(
            'summarize: ' + text.strip(),
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding=False,
        ).to(DEVICE)

        gen_kwargs = _build_generate_kwargs(max_new_tokens, min_new_tokens, length_penalty)
        with torch.inference_mode():
            ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **gen_kwargs,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"Error single chunk: {e}")
        return ""
    finally:
        del inputs, ids


def _summarize_chunks_batched(
    chunks: List[str],
    tokenizer,
    model,
    max_new_tokens: int = 80,
    min_new_tokens: int = 25,
    length_penalty: float = 1.5,
    batch_size: int = SUMMARIZER_BATCH_SIZE,
) -> List[str]:
    """
    Menjalankan inferensi abstraktif secara batch untuk mengurangi overhead tokenisasi dan generate() di CPU.
    """
    if not chunks:
        return []
    if len(chunks) == 1:
        single = summarize_single_chunk(
            chunks[0],
            tokenizer,
            model,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
        )
        return [single] if single else []

    partial: List[str] = []
    gen_kwargs = _build_generate_kwargs(max_new_tokens, min_new_tokens, length_penalty)

    with torch.inference_mode():
        for i in range(0, len(chunks), max(1, batch_size)):
            batch_texts = ['summarize: ' + c.strip() for c in chunks[i:i + max(1, batch_size)]]
            inputs = None
            ids = None
            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    max_length=512,
                    truncation=True,
                    padding=True,
                ).to(DEVICE)
                ids = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    **gen_kwargs,
                )
                for row in ids:
                    decoded = tokenizer.decode(row, skip_special_tokens=True).strip()
                    if decoded:
                        partial.append(decoded)
            finally:
                del inputs, ids

    return partial


def generate_hybrid_summary(
    text: str,
    target_mode: str = "standar",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Dict[str, Any], bool, Optional[str]]:
    """
    Peringkasan HYBRID EXTRACTIVE-ABSTRACTIVE dengan IndoNanoT5-base yang dioptimasi:
    1. Abstractive: Chunking -> batched T5 generate (torch.inference_mode) -> ringkas core abstractive
       (melewati pass kedua jika teks hanya terdiri dari 1 chunk)
    2. Extractive: Ekstraksi kalimat kunci representatif
    3. Merging: Gabung dengan pengecekan overlap (<50%) dan kontrol batas panjang kata.

    Returns:
        (summary_text, metadata_dict, success, error_message)
    """
    if not text or len(text.split()) < 30:
        return (
            text if text else "",
            {"word_count": len(text.split()) if text else 0, "elapsed_time_sec": 0.0},
            False,
            "Teks terlalu pendek untuk diringkas (minimal 30 kata diperlukan)."
        )

    if target_mode == "singkat":
        min_words, max_words = 60, 130
        extractive_sentences = 2
    elif target_mode == "terperinci":
        min_words, max_words = 150, 260
        extractive_sentences = 4
    else:  # standar
        min_words, max_words = SUMMARY_MIN_WORDS, SUMMARY_MAX_WORDS
        extractive_sentences = 3

    start_time = time.time()
    with INFERENCE_LOCK:
        try:
            if progress_callback:
                progress_callback("Memuat model LazarusNLP/IndoNanoT5-base...")
            tokenizer, model = load_summarizer_model()

            # === BAGIAN 1: Abstractive (T5) ===
            chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            if progress_callback:
                progress_callback(f"Memproses {len(chunks)} potongan teks dengan model T5...")

            if len(chunks) == 1:
                # Hemat 1 pass penuh untuk teks <= 1 chunk
                abstractive_core = summarize_single_chunk(
                    chunks[0],
                    tokenizer,
                    model,
                    max_new_tokens=110,
                    min_new_tokens=35,
                    length_penalty=1.5,
                )
            else:
                partial = _summarize_chunks_batched(
                    chunks,
                    tokenizer,
                    model,
                    max_new_tokens=80,
                    min_new_tokens=25,
                    length_penalty=1.5,
                    batch_size=SUMMARIZER_BATCH_SIZE,
                )

                if not partial:
                    fallback = extract_key_sentences(text, n=5)
                    elapsed = round(time.time() - start_time, 2)
                    return fallback, {"word_count": len(fallback.split()), "elapsed_time_sec": elapsed, "method": "extractive_fallback"}, True, None

                combined = ' '.join(partial)
                combined_truncated = ' '.join(combined.split()[:350])
                abstractive_core = summarize_single_chunk(
                    combined_truncated,
                    tokenizer,
                    model,
                    max_new_tokens=110,
                    min_new_tokens=35,
                    length_penalty=1.5,
                )
                if not abstractive_core:
                    abstractive_core = partial[0]

            if not abstractive_core:
                fallback = extract_key_sentences(text, n=5)
                elapsed = round(time.time() - start_time, 2)
                return fallback, {"word_count": len(fallback.split()), "elapsed_time_sec": elapsed, "method": "extractive_fallback"}, True, None

            # === BAGIAN 2: Extractive (Kalimat Kunci Representatif) ===
            if progress_callback:
                progress_callback("Mengekstrak kalimat kunci kontekstual...")
            key_sents = extract_key_sentences(text, n=extractive_sentences)

            # === BAGIAN 3: Penggabungan Cerdas (De-duplication) ===
            abs_words = set(abstractive_core.lower().split())
            final_key = []
            for sent in key_sents.split('. '):
                sent = sent.strip().rstrip('.')
                if not sent:
                    continue
                sent_words = set(sent.lower().split())
                overlap = len(abs_words & sent_words) / max(len(sent_words), 1)
                if overlap < 0.5:
                    final_key.append(sent)

            if final_key:
                final_summary = abstractive_core.rstrip('.') + '. ' + '. '.join(final_key) + '.'
            else:
                final_summary = abstractive_core

            final_summary = re.sub(r'\s+', ' ', final_summary).strip()

            # Validasi batas panjang ringkasan
            n_sum = len(final_summary.split())
            if n_sum < min_words:
                extra = extract_key_sentences(text, n=3)
                if extra:
                    final_summary = final_summary.rstrip('.') + '. ' + extra
                    final_summary = re.sub(r'\s+', ' ', final_summary).strip()
            elif n_sum > max_words:
                words = final_summary.split()[:max_words]
                truncated = ' '.join(words)
                last_punct = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                final_summary = truncated[:last_punct + 1] if last_punct > 50 else truncated + '.'

            if final_summary and final_summary[-1] not in '.!?':
                final_summary += '.'

            elapsed = round(time.time() - start_time, 2)
            n_final_words = len(final_summary.split())

            meta = {
                "word_count": n_final_words,
                "elapsed_time_sec": elapsed,
                "chunks_processed": len(chunks),
                "target_mode": target_mode,
                "method": "hybrid_extractive_abstractive",
            }
            return final_summary, meta, True, None

        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fallback = extract_key_sentences(text, n=5)
            elapsed = round(time.time() - start_time, 2)
            return fallback, {"word_count": len(fallback.split()), "elapsed_time_sec": elapsed, "method": "extractive_oom_fallback"}, True, None
        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            return "", {"word_count": 0, "elapsed_time_sec": elapsed}, False, f"Terjadi kesalahan saat perangkuman: {e}"
        finally:
            force_gc()
