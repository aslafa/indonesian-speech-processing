import re
from typing import Tuple, Dict, Any, List

# Pola filler words bahasa Indonesia non-formal sesuai Cell 13 & 25 di notebook
FILLER_PATTERNS = [
    r'\bee+\b',           # ee, eee
    r'\bum+m*\b',         # um, umm, ummm
    r'\beh+\b',           # eh, ehh
    r'\bah+\b',           # ah (standalone)
    r'\boh+\b',           # oh (standalone)
    r'\banu\b',           # anu
    r'\bhmm+\b',          # hmm, hmmm
    r'\bya kan\b',
    r'\bkan ya\b',
    r'\bgitu loh\b',
    r'\bgitu deh\b',
    r'\bpokoknya\b',
    r'\bmaksudnya\b',
    r'\bsemacam itu\b',
    r'\bapaya\b',
]

# Stopwords bahasa Indonesia untuk ekstraksi kalimat kunci (Cell 17 & 24)
ID_STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk', 'pada',
    'dalam', 'adalah', 'atau', 'juga', 'sudah', 'bisa', 'akan', 'ada', 'aja',
    'sih', 'tuh', 'deh', 'loh', 'gitu', 'kayak', 'tapi', 'jadi', 'kalau', 'karena',
    'emang', 'udah', 'banget', 'buat', 'sama', 'kita', 'gua', 'ya', 'nih', 'terus',
    'kalo', 'gimana', 'punya', 'sebenarnya', 'memang', 'pasti', 'soal', 'kayaknya',
    'mungkin', 'gue', 'lu', 'lo', 'nggak', 'ga', 'udh'
}

def clean_text_id(text: str) -> str:
    """
    Membersihkan teks transkripsi percakapan bahasa Indonesia non-formal:
    1. Mengubah ke lowercase
    2. Menghapus filler words non-formal
    3. Menormalisasi huruf vokal berulang (misal: bangeeet -> banget)
    4. Menghapus karakter non-alfanumerik di luar tanda baca penting
    5. Menormalisasi spasi dan tanda baca
    """
    if not text or not isinstance(text, str):
        return ""
    
    text = text.lower()
    for pat in FILLER_PATTERNS:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
        
    # Normalisasi huruf vokal berulang (>= 3 karakter berulang menjadi 1): bangeeet → banget
    text = re.sub(r'([aeiou])\1{2,}', r'\1', text)
    
    # Hapus karakter non-alfanumerik (pertahankan tanda baca penting)
    text = re.sub(r'[^\w\s.,!\'\-]', ' ', text)
    
    # Normalisasi spasi dan perapatan tanda baca
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,!])', r'\1', text)
    
    return text

def get_cleaning_statistics(raw_text: str, cleaned_text: str) -> Dict[str, Any]:
    """
    Menghitung statistik perbandingan sebelum vs sesudah pembersihan filler.
    """
    raw_words = len(raw_text.split()) if raw_text else 0
    clean_words = len(cleaned_text.split()) if cleaned_text else 0
    removed_words = max(0, raw_words - clean_words)
    reduction_percentage = round((removed_words / raw_words) * 100, 2) if raw_words > 0 else 0.0

    return {
        "raw_word_count": raw_words,
        "clean_word_count": clean_words,
        "removed_filler_count": removed_words,
        "reduction_percentage": reduction_percentage
    }

def _split_pseudo_sentences(text: str, target_words: int = 22) -> List[str]:
    """
    Memecah transkrip panjang tanpa tanda baca akhir kalimat (titik/tanda tanya/seru)
    menjadi klausa/kalimat semu berbasis koma atau batas kata agar ekstraksi kalimat kunci tetap bekerja.
    """
    comma_parts = [p.strip() for p in re.split(r'[,;]+', text) if p.strip()]
    pseudo = []
    buf: List[str] = []
    for part in comma_parts:
        words = part.split()
        if len(buf) + len(words) <= target_words:
            buf.extend(words)
        else:
            if len(buf) > 10:
                pseudo.append(' '.join(buf))
                buf = list(words)
            else:
                buf.extend(words)
                if len(buf) > 10:
                    pseudo.append(' '.join(buf))
                    buf = []
    if len(buf) > 10:
        pseudo.append(' '.join(buf))
    elif buf and pseudo:
        pseudo[-1] = pseudo[-1] + ' ' + ' '.join(buf)

    # Fallback jika koma juga jarang (1 segmen terlalu panjang)
    final_sents: List[str] = []
    for s in pseudo:
        w = s.split()
        if len(w) <= target_words * 2:
            final_sents.append(s)
        else:
            for i in range(0, len(w), target_words):
                chunk = w[i:i + target_words]
                if len(chunk) > 8:
                    final_sents.append(' '.join(chunk))
    return final_sents


def extract_key_sentences(text: str, n: int = 3) -> str:
    """
    Ekstraksi n kalimat paling representatif dari teks transkrip menggunakan skoring frekuensi kata.
    Mendukung transkrip bertanda baca maupun transkrip lisan minim tanda baca titik.
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 10]

    # Jika transkrip panjang namun hampir tidak memiliki tanda baca titik (misal output Whisper small)
    if len(sentences) <= 2 and len(text.split()) > 60:
        pseudo = _split_pseudo_sentences(text)
        if len(pseudo) > len(sentences):
            sentences = pseudo

    if not sentences:
        return ''
    if len(sentences) <= n:
        return '. '.join(sentences) + '.'

    word_freq = {}
    for sent in sentences:
        for w in sent.lower().split():
            if w not in ID_STOPWORDS and len(w) > 3:
                word_freq[w] = word_freq.get(w, 0) + 1

    sent_scores = []
    for i, sent in enumerate(sentences):
        score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
        if i < len(sentences) * 0.2:
            score *= 1.2  # Bobot tambahan untuk pengantar/topik di awal
        sent_scores.append((score, i, sent))

    top = sorted(sent_scores, reverse=True)[:n]
    ordered = sorted(top, key=lambda x: x[1])
    return '. '.join([s for _, _, s in ordered]) + '.'
