"""
AI Music Detection — web demo
Lets anyone try the CNN detector from the TFG "Benchmarking AI-Generated Music
Detection Under TikTok Audio Processing Conditions", choosing which of the
trained checkpoints to run inference with, without installing anything locally.
"""
import os
import subprocess
import tempfile
import logging
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from flask import Flask, request, jsonify, render_template
import imageio_ffmpeg

from model import SimpleSpectrogramCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

RELEASE_BASE = "https://github.com/ainagutierrez/AI-tiktok-detection/releases/download/trained_models"

SR = 16000
DURATION = 3          # seconds — must match training config
N_MELS = 64

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()  # bundled static binary, no system install needed


def normalize_to_wav(input_path):
    """Convert ANY input (audio or video, any codec) into a clean mono 16kHz WAV
    using the bundled ffmpeg binary. This is the only audio decoding path in the
    app — deliberately avoids librosa, whose numba JIT-compiles on first use and
    can blow past the request timeout on a slow/free-tier host."""
    wav_path = input_path + "_norm.wav"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SR),
        "-f", "wav",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0 or not os.path.exists(wav_path):
        raise RuntimeError(
            "No s'ha pogut llegir l'àudio del fitxer "
            f"(ffmpeg: {result.stderr.decode(errors='ignore')[-400:]})"
        )
    return wav_path

extract_audio_from_video = normalize_to_wav

# Every entry below is a REAL checkpoint trained for this thesis, reusing the
# exact SimpleSpectrogramCNN architecture — only the training data differs.
MODELS = {
    "tiktok_700": {
        "label": "CNN — el meu dataset de TikTok (700 mostres)",
        "file": "700_model_tiktok.pth",
        "description": "Entrenada amb un dataset amb només àudios de TikTok (variant de 700 mostres deepfake).",
    },
    "tiktok_600": {
        "label": "CNN — el meu dataset de TikTok (600 mostres)",
        "file": "600_model_tiktok.pth",
        "description": "Entrenada amb un dataset amb només àudios de TikTok (variant de 600 mostres deepfake)",
    },
    "sonics": {
        "label": "CNN — entrenada amb l'enfocament SONICS",
        "file": "model_sonics.pth",
        "description": "CNN entrenada amb les dades de SONICS.",
    },
    "lara_cros_vila": {
        "label": "CNN — entrenada amb l'enfocament Laura Cros-Vila",
        "file": "model_laura_cros_vila.pth",
        "description": "CNN entrenada amb les dades de Cros Vila et al. (embeddings CLAP).",
    },
}

_loaded_models = {}  # cache: key -> torch model in memory


def get_model(key):
    if key not in MODELS:
        raise ValueError(f"Unknown model '{key}'")
    if key in _loaded_models:
        return _loaded_models[key]

    weight_path = os.path.join(WEIGHTS_DIR, MODELS[key]["file"])
    if not os.path.exists(weight_path):
        url = f"{RELEASE_BASE}/{MODELS[key]['file']}"
        logger.info("Downloading weights for '%s' from %s", key, url)
        urllib.request.urlretrieve(url, weight_path)

    model = SimpleSpectrogramCNN(n_classes=1)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _loaded_models[key] = model
    logger.info("Model '%s' loaded and cached", key)
    return model


def load_audio(path):
    """Load audio via soundfile only — fast, no JIT compilation, no surprises."""
    import soundfile as sf
    waveform, sr = sf.read(path, dtype="float32")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return torch.from_numpy(waveform).float(), sr


def predict(model, audio_path):
    # Always normalize first: guarantees mono + SR sample rate + a format
    # soundfile can read, regardless of what the user actually uploaded.
    normalized_path = normalize_to_wav(audio_path)
    try:
        waveform, sr = load_audio(normalized_path)
    finally:
        try:
            os.unlink(normalized_path)
        except OSError:
            pass

    num_samples = SR * DURATION
    if waveform.shape[0] > num_samples:
        # center crop, more representative than the very start of the clip
        start = (waveform.shape[0] - num_samples) // 2
        waveform = waveform[start:start + num_samples]
    else:
        waveform = F.pad(waveform, (0, num_samples - waveform.shape[0]))

    mel = T.MelSpectrogram(sample_rate=SR, n_mels=N_MELS)
    spec = torch.log(mel(waveform) + 1e-6).unsqueeze(0).unsqueeze(0)
    spec = (spec - spec.mean()) / (spec.std() + 1e-9)

    with torch.no_grad():
        logit = model(spec)
        prob_fake = torch.sigmoid(logit).item()

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    return label, prob_fake


@app.route("/")
def index():
    options = [{"key": k, **v} for k, v in MODELS.items()]
    return render_template("index.html", models=options)


@app.route("/api/models")
def api_models():
    return jsonify([{"key": k, "label": v["label"], "description": v["description"]} for k, v in MODELS.items()])


@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No s'ha rebut cap fitxer d'àudio."}), 400

    model_key = request.form.get("model", "tiktok_700")
    if model_key not in MODELS:
        return jsonify({"success": False, "error": f"Model desconegut: {model_key}"}), 400

    upload_file = request.files["audio"]
    suffix = os.path.splitext(upload_file.filename or "upload.wav")[1].lower() or ".wav"
    is_video = suffix in VIDEO_EXTENSIONS

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        upload_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        model = get_model(model_key)
        label, prob_fake = predict(model, tmp_path)  # predict() normalizes via ffmpeg internally
        return jsonify({
            "success": True,
            "model": model_key,
            "model_label": MODELS[model_key]["label"],
            "label": label,
            "probability_fake": prob_fake,
            "source": "video" if is_video else "audio",
        })
    except Exception as e:
        logger.exception("Detection failed")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
