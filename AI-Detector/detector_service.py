# detector_service.py
"""
Prototype: AI-Generated Phishing Email and Deepfake Media Detector
Single-file Flask service (prototype).
Endpoints:
 - POST /scan_text     -> JSON { "text": "..."}  => returns text-based phishing/impersonation score + reasons
 - POST /scan_email    -> JSON { "subject":"", "body":"", "from":"", "headers":{}} => returns phishing score
 - POST /scan_file     -> multipart/form-data file=... => auto-detects mime and run detectors (image/video/audio/pdf)
Notes:
 - This is a prototype. Replace placeholder models with production-grade models/datasets.
 - For videos: extracts frames and runs image model on faces/frames and aggregates scores.
"""

import io
import os
import tempfile
import json
from typing import Dict, Any, List
from flask import Flask, request, jsonify
import numpy as np

# --- ML / CV / Audio imports ---
import re
from PIL import Image
import pytesseract
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import torchaudio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# --- Setup Flask ---
app = Flask(__name__)

# ---------- Utility helpers ----------
def safe_read_file(storage_file):
    data = storage_file.read()
    return data

def simple_text_cleanup(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# ---------- Text phishing / imitation detector (simple TF-IDF + LR) ----------
class TextPhishDetector:
    """
    Simple TF-IDF + LogisticRegression classifier prototype.
    - If you have datasets, train and save model with train_text_model() utility below.
    - In production, use larger datasets + transformer-based classifiers (e.g., fine-tuned RoBERTa).
    """
    def __init__(self, model_path=None):
        self.model_path = model_path or "text_phish_clf.joblib"
        if os.path.exists(self.model_path):
            obj = joblib.load(self.model_path)
            self.vectorizer = obj['vectorizer']
            self.clf = obj['clf']
        else:
            # initialize empty (untrained) - will behave conservatively (low-confidence).
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
            self.clf = LogisticRegression()
            # Not fitted

    def predict_proba(self, text: str) -> float:
        text = simple_text_cleanup(text)
        try:
            X = self.vectorizer.transform([text])
            p = self.clf.predict_proba(X)[0][1]
            return float(p)
        except Exception:
            # model not trained or error -> fallback heuristics
            return float(self.heuristic_score(text))

    def heuristic_score(self, text: str) -> float:
        # simple heuristics: presence of "urgent", mismatched domains, login link patterns, base64 images, spoofing words
        score = 0.0
        text_l = text.lower()
        heuristics = [
            (r'urgent|immediate action|verify (your|account)|update your', 0.25),
            (r'click here|http[s]?://\S+|login here|verify account', 0.2),
            (r'password|account suspended|limited time', 0.15),
            (r'bitcoin|wire transfer|pay now', 0.15),
            (r'dear (customer|user|member)', 0.05),
        ]
        for patt, w in heuristics:
            if re.search(patt, text_l):
                score += w
        return min(score, 0.99)

    @staticmethod
    def train_text_model(X_texts: List[str], y: List[int], save_path="text_phish_clf.joblib"):
        """
        Train and save a TF-IDF + LogisticRegression classifier.
        X_texts: list of text (emails, bodies)
        y: binary labels (1=phishing/fake, 0=legit)
        """
        vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
        X = vectorizer.fit_transform(X_texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        joblib.dump({'vectorizer': vectorizer, 'clf': clf}, save_path)
        print("Saved text model to", save_path)

# instantiate
text_detector = TextPhishDetector()

# ---------- Logo detector (OpenCV template matching) ----------
class LogoDetector:
    """
    Simple template-matching logo detector.
    Put logo templates in a folder and it will try to match via cv2.matchTemplate.
    Limitations: scale/rotation sensitive. Replace with object detector (YOLO/SSD/Faster-RCNN) for production.
    """
    def __init__(self, templates_dir="logo_templates"):
        self.templates = []
        self.templates_dir = templates_dir
        if os.path.isdir(templates_dir):
            for fname in os.listdir(templates_dir):
                p = os.path.join(templates_dir, fname)
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates.append((os.path.basename(fname), img))
        # else: no templates yet

    def match(self, img_bgr) -> List[Dict[str,Any]]:
        results = []
        h_img, w_img = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        for name, tpl in self.templates:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.75: # threshold
                results.append({
                    "template": name,
                    "score": float(max_val),
                    "location": max_loc
                })
        return results

logo_detector = LogoDetector()

# ---------- Image/Video deepfake detector (MobileNetV2 scaffold) ----------
class ImageDeepfakeDetector:
    """
    Prototype image classifier using MobileNetV2 backbone. Replace with stronger model (Xception/efficientnet) fine-tuned on deepfake datasets.
    """
    def __init__(self, device='cpu', model_path=None):
        self.device = device
        self.model_path = model_path
        # Use pretrained mobilenetv2 and replace classifier for binary output (0=real,1=fake)
        self.model = models.mobilenet_v2(pretrained=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2)
        )
        self.model = self.model.to(device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict_image(self, pil_image: Image.Image) -> float:
        img = pil_image.convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0,1].item()
        return float(probs)

    def predict_on_frames(self, frames: List[Image.Image]) -> Dict[str,Any]:
        probs = [self.predict_image(f) for f in frames]
        return {
            "per_frame": probs,
            "avg_score": float(np.mean(probs)),
            "max_score": float(np.max(probs))
        }

image_detector = ImageDeepfakeDetector(device='cpu')

def extract_frames_from_video(video_path, max_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total>0 else 1
    i = 0
    grabbed = 0
    while grabbed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            # convert BGR->PIL
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil)
            grabbed += 1
        i += 1
    cap.release()
    return frames

# ---------- Audio deepfake detector (spectrogram CNN scaffold) ----------
class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

class AudioFakeDetector:
    """
    Prototype: converts audio to mel-spectrogram and runs a tiny CNN.
    Replace with ECAPA-TDNN or Wav2Vec2-based detector for production.
    """
    def __init__(self, model_path=None, device='cpu'):
        self.model = SimpleAudioCNN().to(device)
        self.device = device
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=64)

    def predict(self, waveform: torch.Tensor, sr: int) -> float:
        # Resample if needed
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
            sr = 16000
        if waveform.ndim>1:
            waveform = waveform.mean(dim=0, keepdim=True)
        spec = self.melspec(waveform)  # (n_mels, T)
        spec = torch.log1p(spec)
        spec = spec.unsqueeze(0).to(self.device)  # shape (1,1,n_mels,T)
        with torch.no_grad():
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=1)[0,1].item()
        return float(probs)

audio_detector = AudioFakeDetector()

# ---------- Document detector (OCR + text classifier) ----------
class DocumentDetector:
    """
    Runs OCR (pytesseract) and sends extracted text to text detector.
    """
    def __init__(self, text_detector):
        self.text_detector = text_detector

    def analyze_image_doc(self, image_pil: Image.Image) -> Dict[str,Any]:
        text = pytesseract.image_to_string(image_pil)
        phish_score = self.text_detector.predict_proba(text)
        return {"ocr_text": text, "phish_score": phish_score}

doc_detector = DocumentDetector(text_detector)

# ---------- Aggregation and API endpoints ----------
def analyze_image_file(bgr_img) -> Dict[str,Any]:
    # logo detection
    logos = logo_detector.match(bgr_img)
    # image deepfake
    pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    img_score = image_detector.predict_image(pil)
    return {"logo_matches": logos, "image_deepfake_score": img_score}

@app.route("/scan_text", methods=["POST"])
def scan_text():
    payload = request.get_json(force=True)
    text = payload.get("text","")
    text = simple_text_cleanup(text)
    prob = text_detector.predict_proba(text)
    reasons = []
    if prob > 0.7:
        reasons.append("High phishing/impersonation signal from text patterns.")
    elif prob > 0.3:
        reasons.append("Moderate risk: suspicious tokens/phrases.")
    else:
        reasons.append("Low textual risk detected.")
    return jsonify({"phish_probability": prob, "reasons": reasons})

@app.route("/scan_email", methods=["POST"])
def scan_email():
    payload = request.get_json(force=True)
    subject = payload.get("subject","")
    body = payload.get("body","")
    from_header = payload.get("from","")
    full_text = f"subject: {subject}\nfrom: {from_header}\n\n{body}"
    prob = text_detector.predict_proba(full_text)
    meta = {"subject": subject, "from": from_header}
    # quick heuristics: mismatched domain vs from header (very basic)
    domain_mismatch = False
    urls = re.findall(r'https?://[^\s]+', full_text)
    if urls and from_header:
        from_domain = re.findall(r'@([A-Za-z0-9.-]+)', from_header)
        if from_domain:
            from_dom = from_domain[0].lower()
            for u in urls:
                if from_dom not in u.lower():
                    domain_mismatch = True
    if domain_mismatch:
        prob = min(0.99, prob + 0.15)
    return jsonify({"phish_probability": prob, "meta": meta, "domain_mismatch": domain_mismatch})

@app.route("/scan_file", methods=["POST"])
def scan_file():
    """
    Accepts multipart form 'file'.
    Auto-detects mime type via python-magic (if available) else uses extension and OpenCV fallback.
    """
    if 'file' not in request.files:
        return jsonify({"error":"no file part"}), 400
    f = request.files['file']
    filename = f.filename or "upload"
    data = f.read()
    # temp save
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    try:
        # try audio read
        try:
            waveform, sr = torchaudio.load(tmp.name)
            # if read succeeds, treat as audio
            audio_score = audio_detector.predict(waveform, sr)
            os.unlink(tmp.name)
            return jsonify({"type":"audio", "audio_fake_score": audio_score})
        except Exception:
            pass

        # try video via cv2
        try:
            cap = cv2.VideoCapture(tmp.name)
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                # video file
                frames = extract_frames_from_video(tmp.name, max_frames=12)
                img_results = image_detector.predict_on_frames(frames)
                # also logo-check first frame
                first_bgr = cv2.cvtColor(np.array(frames[0]), cv2.COLOR_RGB2BGR)
                logo_results = logo_detector.match(first_bgr)
                os.unlink(tmp.name)
                return jsonify({
                    "type":"video",
                    "frame_analysis": img_results,
                    "logo_matches": logo_results
                })
        except Exception:
            pass

        # try image
        try:
            pil = Image.open(io.BytesIO(data))
            bgr = cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)
            img_results = analyze_image_file(bgr)
            # run document OCR too
            doc_res = doc_detector.analyze_image_doc(pil)
            os.unlink(tmp.name)
            return jsonify({"type":"image", "image_results": img_results, "document_results": doc_res})
        except Exception:
            pass

        # fallback: try PDF -> render first page with poppler (not included) or use OCR on raw bytes
        # Simple fallback: attempt to decode as text (for plain docs)
        try:
            txt = data.decode('utf-8', errors='ignore')
            txt_score = text_detector.predict_proba(txt)
            os.unlink(tmp.name)
            return jsonify({"type":"text_file", "text_phish_score": txt_score})
        except Exception:
            os.unlink(tmp.name)
            return jsonify({"error":"unknown file type or unsupported"}, 400)
    finally:
        if os.path.exists(tmp.name):
            try: os.unlink(tmp.name)
            except: pass

if __name__ == "__main__":
    # dev server
    app.run(host="0.0.0.0", port=5000, debug=True)
