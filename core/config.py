import os
import shutil
import torch

# Deteksi ketersediaan GPU CUDA vs CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Optimasi CPU Threading (Optimal untuk CPU hybrid seperti Intel Core i7-1355U: 10 core / 12 thread)
CPU_COUNT = os.cpu_count() or 4
OPTIMAL_CPU_THREADS = min(6, max(1, CPU_COUNT // 2))

# Model constants
WHISPER_AVAILABLE_MODELS = ["small", "medium"]
DEFAULT_WHISPER_MODEL = "small"  # Cepat di CPU, opsi beralih ke medium tersedia di UI
SUMMARIZER_MODEL_NAME = "LazarusNLP/IndoNanoT5-base"

# Parameter Transkripsi Faster-Whisper (Hasil eksperimen benchmark CPU)
WHISPER_CPU_THREADS = OPTIMAL_CPU_THREADS
WHISPER_NUM_WORKERS = 1
WHISPER_BEAM_SIZE = 1
WHISPER_BEST_OF = 1
WHISPER_LANGUAGE = "id"
WHISPER_CONDITION_ON_PREVIOUS_TEXT = True
WHISPER_WORD_TIMESTAMPS = False
WHISPER_VAD_FILTER = True
WHISPER_VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 250,
}

# Chunking & Summarization Parameters (Dioptimalkan untuk CPU + IndoNanoT5-base)
CHUNK_SIZE = 350
CHUNK_OVERLAP = 50
SUMMARY_MIN_WORDS = 80
SUMMARY_MAX_WORDS = 250
SUMMARIZER_NUM_BEAMS = 1
SUMMARIZER_BATCH_SIZE = 4
SUMMARIZER_REPETITION_PENALTY = 1.5
SUMMARIZER_NO_REPEAT_NGRAM_SIZE = 3
TORCH_CPU_THREADS = OPTIMAL_CPU_THREADS

# Batas keamanan RAM sistem untuk perlindungan OOM (persentase RAM terpakai / sisa minimal MB)
RAM_HIGH_WATERMARK_PCT = 86.0
RAM_MIN_AVAILABLE_MB = 2000.0

# Direktori Proyek, Temporary, Cache, dan Demo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "temp_data")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache_data")
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo_data")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DEMO_DIR, exist_ok=True)

def find_ffmpeg_bin_dir():
    """
    Mencari folder bin yang memuat executable ffmpeg (dan ffprobe).
    Mendukung deteksi otomatis multi-platform & multi-sumber:
    1. Variabel lingkungan: FFMPEG_BIN, FFMPEG_DIR, atau FFMPEG_PATH
    2. PATH sistem saat ini (jika ffmpeg dan ffprobe sudah terdaftar)
    3. Paket WinGet Windows (%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\*ffmpeg*\\...\\bin)
    4. Lokasi standar instalasi Windows (Program Files, Chocolatey, Scoop, C:\\ffmpeg\\bin)
    5. Folder lokal proyek (bin atau ffmpeg\\bin)
    6. Fallback PATH sistem jika hanya ffmpeg yang terdeteksi
    7. Fallback biner imageio_ffmpeg
    
    Returns:
        tuple (bin_dir: Optional[str], ffmpeg_path: Optional[str], ffprobe_path: Optional[str])
    """
    # 1. Cek variabel lingkungan kustom
    for env_var in ("FFMPEG_BIN", "FFMPEG_DIR", "FFMPEG_PATH"):
        val = os.environ.get(env_var)
        if val:
            candidate_dir = val if os.path.isdir(val) else os.path.dirname(val)
            if os.path.isdir(candidate_dir):
                ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
                ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
                f_exe = os.path.join(candidate_dir, ffmpeg_name)
                p_exe = os.path.join(candidate_dir, ffprobe_name)
                if os.path.isfile(f_exe):
                    return candidate_dir, f_exe, (p_exe if os.path.isfile(p_exe) else None)

    # 2. Cek apakah ffmpeg dan ffprobe sudah ada di PATH sistem saat ini
    ffmpeg_in_path = shutil.which("ffmpeg")
    ffprobe_in_path = shutil.which("ffprobe")
    if ffmpeg_in_path and ffprobe_in_path:
        return os.path.dirname(os.path.abspath(ffmpeg_in_path)), ffmpeg_in_path, ffprobe_in_path

    # 3. Cek paket WinGet Windows (%LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\*ffmpeg*\\...\\bin)
    if os.name == "nt":
        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            winget_pkgs = os.path.join(localappdata, "Microsoft", "WinGet", "Packages")
            if os.path.isdir(winget_pkgs):
                try:
                    for entry in os.scandir(winget_pkgs):
                        if entry.is_dir() and "ffmpeg" in entry.name.lower():
                            for root, _, files in os.walk(entry.path):
                                files_lower = {f.lower(): f for f in files}
                                if "ffmpeg.exe" in files_lower:
                                    f_exe = os.path.join(root, files_lower["ffmpeg.exe"])
                                    p_exe = os.path.join(root, files_lower["ffprobe.exe"]) if "ffprobe.exe" in files_lower else None
                                    return root, f_exe, p_exe
                except Exception:
                    pass

    # 4. Cek lokasi standar instalasi Windows
    if os.name == "nt":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        standard_paths = [
            # WinGet Links / WindowsApps
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps"),
            # Program Files
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "ffmpeg", "bin"),
            r"C:\ffmpeg\bin",
            # Chocolatey & Scoop
            os.path.join(os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey"), "bin"),
            os.path.expanduser(r"~\scoop\apps\ffmpeg\current\bin"),
            os.path.expanduser(r"~\scoop\shims"),
            # Folder lokal proyek
            os.path.join(project_root, "bin"),
            os.path.join(project_root, "ffmpeg", "bin"),
        ]
        for p in standard_paths:
            if os.path.isdir(p):
                f_exe = os.path.join(p, "ffmpeg.exe")
                p_exe = os.path.join(p, "ffprobe.exe")
                if os.path.isfile(f_exe):
                    return p, f_exe, (p_exe if os.path.isfile(p_exe) else None)

    # 5. Fallback PATH sistem jika hanya ffmpeg yang terdeteksi
    if ffmpeg_in_path:
        return os.path.dirname(os.path.abspath(ffmpeg_in_path)), ffmpeg_in_path, ffprobe_in_path

    # 6. Fallback biner imageio_ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            bin_dir = os.path.dirname(os.path.abspath(exe))
            if os.name == "nt":
                target_alias = os.path.join(bin_dir, "ffmpeg.exe")
                if not os.path.exists(target_alias):
                    try:
                        shutil.copyfile(exe, target_alias)
                    except Exception:
                        pass
                actual_ffmpeg = target_alias if os.path.exists(target_alias) else exe
            else:
                actual_ffmpeg = exe
            return bin_dir, actual_ffmpeg, None
    except Exception:
        pass

    return None, None, None

def setup_ffmpeg_path():
    """
    Menyiapkan direktori bin FFmpeg ke os.environ['PATH'] runtime aplikasi
    sehingga ffmpeg dan ffprobe dapat ditemukan langsung oleh subprocess dan yt-dlp.
    Returns:
        tuple (bin_dir: Optional[str], ffmpeg_path: Optional[str], ffprobe_path: Optional[str])
    """
    bin_dir, ffmpeg_exe, ffprobe_exe = find_ffmpeg_bin_dir()

    if bin_dir and os.path.isdir(bin_dir):
        current_path = os.environ.get("PATH", "")
        # Normalisasi path untuk menghindari duplikasi
        normalized_parts = [os.path.normcase(os.path.abspath(p)) for p in current_path.split(os.pathsep) if p]
        norm_bin = os.path.normcase(os.path.abspath(bin_dir))
        if norm_bin not in normalized_parts:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path

        # Refresh lookup jika salah satu belum ada
        if not ffprobe_exe:
            ffprobe_exe = shutil.which("ffprobe")
        if not ffmpeg_exe:
            ffmpeg_exe = shutil.which("ffmpeg")

    return bin_dir, ffmpeg_exe, ffprobe_exe

# Setup FFmpeg & FFprobe saat modul dimuat
FFMPEG_BIN_DIR, FFMPEG_PATH, FFPROBE_PATH = setup_ffmpeg_path()
