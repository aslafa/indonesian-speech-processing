import json
from datetime import datetime
from typing import Dict, Any

def create_export_txt(
    raw_transcript: str,
    cleaned_transcript: str,
    summary: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Menghasilkan format teks terstruktur untuk unduhan file .txt.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "LAPORAN TRANSKRIPSI DAN PERANGKUMAN - INDONESIAN SPEECH-TO-TEXT",
        f"Waktu Pemrosesan: {timestamp}",
        "=" * 70,
        "",
        "--- 1. INFORMASI METADATA & PARAMETER ---",
        f"Tipe Sumber        : {metadata.get('source_type', 'N/A')}",
        f"Model Transkripsi  : {metadata.get('whisper_model', 'N/A')}",
        f"Model Peringkas    : {metadata.get('summarizer_model', 'LazarusNLP/IndoNanoT5-base')}",
        f"Durasi Audio       : {metadata.get('audio_duration_sec', 0.0)} detik",
        f"Waktu Transkripsi  : {metadata.get('transcribe_time_sec', 0.0)} detik",
        f"Real-Time Factor   : {metadata.get('rtf', 0.0)} (RTF)",
        f"Waktu Peringkasan  : {metadata.get('summary_time_sec', 0.0)} detik",
        f"Jumlah Kata Mentah : {metadata.get('raw_word_count', 0)} kata",
        f"Jumlah Kata Bersih : {metadata.get('clean_word_count', 0)} kata",
        f"Filler Dihapus     : {metadata.get('removed_filler_count', 0)} kata ({metadata.get('reduction_percentage', 0.0)}%)",
        f"Jumlah Kata Ringkas: {metadata.get('summary_word_count', 0)} kata",
        "",
        "--- 2. RINGKASAN EKSEKUTIF (HYBRID) ---",
        summary if summary else "[Ringkasan tidak tersedia]",
        "",
        "--- 3. TRANSKRIP BERSIH (CLEANED) ---",
        cleaned_transcript if cleaned_transcript else "[Transkrip bersih tidak tersedia]",
        "",
        "--- 4. TRANSKRIP MENTAH (RAW) ---",
        raw_transcript if raw_transcript else "[Transkrip mentah tidak tersedia]",
        "",
        "=" * 70,
        "Dihasilkan oleh Indonesian Speech-to-Text Platform",
        "=" * 70
    ]
    return "\n".join(lines)

def create_export_json(
    raw_transcript: str,
    cleaned_transcript: str,
    summary: str,
    metadata: Dict[str, Any]
) -> str:
    """
    Menghasilkan format JSON terstruktur untuk unduhan file .json.
    """
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "metadata": metadata,
        "summary": summary,
        "cleaned_transcript": cleaned_transcript,
        "raw_transcript": raw_transcript
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)
