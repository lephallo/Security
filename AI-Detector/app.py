# app.py
import os
import logging
import re
import tempfile
import joblib
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from torchvision import models as tv_models, transforms
from PIL import Image
import torchaudio
import numpy as np
import cv2
import torch.nn.functional as F
# ---------------------------
# UTILS
# ---------------------------
def jsonify_safe(data):
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj
    return jsonify(convert(data))

def clean_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def save_temp(file_storage):
    suffix = os.path.splitext(secure_filename(file_storage.filename))[1]
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_FOLDER)
    file_storage.save(tf.name)
    tf.close()
    return tf.name

# ---------------------------
# CONFIG
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODELS_DIR = BASE_DIR

MODEL_FILENAME = "phishing_model.pkl"
VECTORIZER_FILENAME = "vectorizer.pkl"
IMAGE_MODEL_FILENAME = "model.pth"
AUDIO_MODEL_FILENAME = "models/audio_mobilenetv2.pth"

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

ALLOWED_EXTENSIONS = {
    "txt",
    "png", "jpg", "jpeg", "gif", "bmp", "tiff",
    "wav", "mp3", "m4a", "flac",
    "mp4", "mov", "avi", "mkv"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# FLASK SETUP
# ---------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# LOAD MODELS
# ---------------------------
def load_joblib_model(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def _build_imagenet_mobilenet_v2(num_classes: int = 2):
    m = tv_models.mobilenet_v2(pretrained=False)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    return m

def load_torch_model(filename: str):
    path = os.path.join(MODELS_DIR, filename) if not os.path.isabs(filename) else filename
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loaded = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
        class_names = loaded.get("classes", ["real", "fake"])
    else:
        state_dict = loaded
        class_names = ["real", "fake"]

    model = _build_imagenet_mobilenet_v2(num_classes=len(class_names))
    new_state = {k[len("module."):] if k.startswith("module.") else k:v for k,v in state_dict.items()}
    model.load_state_dict(new_state)
    model.eval()
    model.classes = class_names
    return model

# Load models safely
try:
    phishing_model = load_joblib_model(MODEL_FILENAME)
    phishing_vectorizer = load_joblib_model(VECTORIZER_FILENAME)
    logger.info("Text phishing model loaded ✅")
except Exception as e:
    phishing_model = phishing_vectorizer = None
    logger.exception("Failed to load text model: %s", e)

try:
    image_model = load_torch_model(IMAGE_MODEL_FILENAME)
    logger.info("Image/Video model loaded ✅")
except Exception as e:
    image_model = None
    logger.warning("Image/Video model not loaded: %s", e)

try:
    audio_model = load_torch_model(AUDIO_MODEL_FILENAME)
    logger.info("Audio model loaded ✅")
except Exception as e:
    audio_model = None
    logger.warning("Audio model not loaded: %s", e)

# ---------------------------
# AUDIO HELPER
# ---------------------------


def preprocess_audio(waveform, sr, device):
    """
    Convert waveform to 3-channel 224x224 mel-spectrogram tensor
    """
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    # Mel spectrogram -> decibels
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=64
    )(waveform.to(device))
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # Resize to 224x224 and make 3 channels
    mel_spec = F.interpolate(
        mel_spec.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False
    )[0]
    mel_spec = mel_spec.repeat(3,1,1)  # 3 channels

    # Normalize
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(3,1,1)
    mel_spec = (mel_spec - mean) / (std + 1e-6)

    return mel_spec.unsqueeze(0)  # [1,3,224,224]

def predict_audio(file_path, device=torch.device("cpu")):
    """
    Predict real/fake for audio file
    """
    if audio_model is None:
        raise RuntimeError("Audio model not loaded.")

    try:
        waveform, sr = torchaudio.load(file_path, backend="soundfile")
        input_tensor = preprocess_audio(waveform, sr, device).to(device)

        with torch.no_grad():
            logits = audio_model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            label_idx = int(probs.argmax())
            class_names = getattr(audio_model, "classes", ["real","fake"])
            label = class_names[label_idx]
            probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

        return {"prediction": label_idx, "label": label, "probabilities": probabilities}

    except Exception as e:
        logger.exception("Audio prediction failed for %s: %s", file_path, e)
        return {"error": str(e)}

# ---------------------------
# ROUTES
# ---------------------------
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
def home():
    """Serve the homepage"""
    return render_template("home.html") 
@app.route("/mail")
def mail_page():
    """Serve the email detector frontend (mail.html)"""
    return render_template("mail.html")
@app.route("/index")
def index_page():
    """Serve the deepfake media detection frontend (index.html)"""
    return render_template("index.html")

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "service": "AI Multimodal Phishing Detector",
        "text_model": phishing_model is not None,
        "image_model": image_model is not None,
        "audio_model": audio_model is not None,
        "video_model": image_model is not None,
    })

@app.route("/predict", methods=["POST"])
def predict_text():
    if phishing_model is None or phishing_vectorizer is None:
        return jsonify({"error": "Text model not loaded"}), 503
    data = request.get_json(silent=True) or {}
    subject = data.get("subject", "")
    body = data.get("body", "")
    text_input = f"{subject} {body}".strip()
    if not text_input:
        return jsonify({"error": "No text provided"}), 400
    cleaned = clean_text(text_input)
    try:
        X = phishing_vectorizer.transform([cleaned])
        pred = phishing_model.predict(X)[0]
        probs = phishing_model.predict_proba(X)[0].tolist() if hasattr(phishing_model, "predict_proba") else None
        label = "phishing" if str(pred) in ("1", "phishing", "True", "true") else "legitimate"
        return jsonify({"prediction": str(pred), "label": label, "probabilities": probs}), 200
    except Exception as e:
        logger.exception("Text prediction failed: %s", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/scan_email", methods=["POST"])
def scan_email():
    if phishing_model is None or phishing_vectorizer is None:
        return jsonify({"error": "Email model not loaded"}), 503
    subject = request.form.get("subject", "")
    body = request.form.get("body", "")
    if not subject and not body:
        return jsonify({"error": "No email content provided"}), 400
    text_input = f"{subject} {body}".strip()
    cleaned = clean_text(text_input)
    try:
        X = phishing_vectorizer.transform([cleaned])
        pred = phishing_model.predict(X)[0]
        probs = phishing_model.predict_proba(X)[0].tolist() if hasattr(phishing_model, "predict_proba") else None
        label = "phishing" if str(pred) in ("1", "phishing", "True", "true") else "legitimate"
        return jsonify({"prediction": str(pred), "label": label, "probabilities": probs}), 200
    except Exception as e:
        logger.exception("Email prediction failed: %s", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/scan_file", methods=["POST"])
def scan_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "File extension not allowed"}), 400

    saved = save_temp(f)
    ext = os.path.splitext(saved)[1].lower().lstrip(".")

    try:
        # ---------- IMAGE ----------
        if ext in ("png", "jpg", "jpeg", "bmp", "tiff", "gif"):
            if image_model is None:
                return jsonify({"error": "Image model not loaded"}), 503
            img = Image.open(saved).convert("RGB")
            preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            input_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                logits = image_model(input_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                label_idx = int(probs.argmax())
                class_names = getattr(image_model, "classes", ["real","fake"])
                label = class_names[label_idx]
                probabilities = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
            return jsonify({"type":"image", "label":label, "probabilities":probabilities}), 200

        # ---------- VIDEO ----------
        if ext in ("mp4", "mov", "avi", "mkv"):
            if image_model is None:
                return jsonify({"error": "Image/video model not loaded"}), 503
            preprocess = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            cap = cv2.VideoCapture(saved)
            frame_preds = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                input_tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    logits = image_model(input_tensor)
                    pred_idx = int(torch.argmax(logits, dim=1).item())
                    frame_preds.append(pred_idx)
            cap.release()
            if not frame_preds:
                return jsonify({"error": "No frames read from video"}), 400
            counts = np.bincount(frame_preds)
            video_pred_idx = int(np.argmax(counts))
            class_names = getattr(image_model, "classes", ["real","fake"])
            label = class_names[video_pred_idx]
            return jsonify({"type":"video", "label": label, "frame_predictions": frame_preds}), 200

        # ---------- AUDIO ----------
        if ext in ("wav", "mp3", "flac", "m4a"):
            if audio_model is None:
                return jsonify({"error": "Audio model not loaded"}), 503
            try:
                result = predict_audio(saved)
                return jsonify({"type": "audio", **result}), 200
            except Exception as e:
                return jsonify({"type": "audio", "error": "Audio prediction failed", "detail": str(e)}), 500

        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    finally:
        try:
            os.remove(saved)
        except Exception:
            pass

# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
