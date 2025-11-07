# train_email_with_attachments.py
"""
Extended training script for phishing email detection.
Modifications:
 - Reads possible attachment paths from CSV column 'attachment_path' (optional).
 - Extracts text from attachments (txt, log, pdf, docx) and appends to email text for TF-IDF.
 - (Optional) Extracts images from PDF/attachments and computes simple image embedding features
   using a pretrained MobileNetV2 as a feature extractor (if torch + torchvision available).
 - Adds attachment-derived numeric features:
     - attachment_text_len (words)
     - num_images
     - avg_image_emb_norm
 - Combines text TF-IDF with numeric features and trains LinearSVC as before.
 - Saves model, tfidf vectorizer, and numeric scaler.
"""

import os
import sys
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from scipy.sparse import hstack, csr_matrix

# ---------- Optional libs for attachments / images ----------
USE_PYMUPDF = False
USE_PYTORCH = False
USE_PYTESSERACT = False
USE_DOCX = False

try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
except Exception:
    print("PyMuPDF (fitz) not available — PDF image/text extraction disabled.", file=sys.stderr)

try:
    from PIL import Image
    import io
    import torch
    import torchvision.transforms as T
    import torchvision.models as models
    USE_PYTORCH = True
except Exception:
    print("torch/torchvision/Pillow not available — image embedding features disabled.", file=sys.stderr)

try:
    import pytesseract
    USE_PYTESSERACT = True
except Exception:
    print("pytesseract not available — OCR text from images disabled.", file=sys.stderr)

try:
    from docx import Document
    USE_DOCX = True
except Exception:
    print("python-docx not available — .docx text extraction disabled.", file=sys.stderr)

# ---------------- CONFIG ----------------
DATA_CSV = "emails_dataset.csv"      # your CSV
MODEL_FILE = "phishing_email_model_with_attachments.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SCALER_FILE = "numeric_scaler.pkl"

# If torch is available, build an image feature extractor
IMAGE_EMBED_DIM = None
image_model = None
image_preprocess = None
USE_IMAGE_MODEL = USE_PYTORCH  # set to False if you don't want image model even if torch is present

if USE_IMAGE_MODEL:
    try:
        # MobileNetV2 backbone: we'll use features + global pooling to get embeddings
        img_m = models.mobilenet_v2(pretrained=True)
        # remove classifier to use features; create a simple feature extractor
        # We'll run images through img_m.features then global avg pool on the resulting tensor
        img_m = img_m.eval()
        image_model = img_m.features  # a nn.Sequential of feature extractor
        # compute embedding dim by doing a forward pass with dummy
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = image_model(dummy)
            # feats shape [1, C, H, W]; do global avg pool:
            emb = feats.mean(dim=[2,3])  # avg pool
            IMAGE_EMBED_DIM = emb.shape[1]
        image_preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        print(f"Image model loaded: embed dim = {IMAGE_EMBED_DIM}")
    except Exception as e:
        print("Failed to initialize image model:", e, file=sys.stderr)
        USE_IMAGE_MODEL = False
        image_model = None

# ---------------- helpers ----------------
def extract_text_from_pdf(path: Path):
    """Extracts textual content from PDF using PyMuPDF (if available)."""
    if not USE_PYMUPDF:
        return ""
    try:
        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            try:
                t = page.get_text()
                if t:
                    texts.append(t)
            except Exception:
                pass
        doc.close()
        return "\n".join(texts)
    except Exception:
        return ""

def extract_images_from_pdf(path: Path):
    """Return list of PIL.Image objects extracted from PDF (if PyMuPDF available)."""
    imgs = []
    if not USE_PYMUPDF:
        return imgs
    try:
        doc = fitz.open(str(path))
        for page in doc:
            for imginfo in page.get_images(full=True):
                xref = imginfo[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                try:
                    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    imgs.append(im)
                except Exception:
                    pass
        doc.close()
    except Exception:
        pass
    return imgs

def extract_text_from_docx(path: Path):
    if not USE_DOCX:
        return ""
    try:
        doc = Document(str(path))
        texts = [p.text for p in doc.paragraphs if p.text]
        return "\n".join(texts)
    except Exception:
        return ""

def extract_text_from_txt(path: Path):
    try:
        with open(path, "r", encoding="utf8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def ocr_image_text(pil_img):
    if not USE_PYTESSERACT:
        return ""
    try:
        return pytesseract.image_to_string(pil_img)
    except Exception:
        return ""

def compute_image_embedding(pil_img):
    """Return 1-D numpy embedding vector or None if not available."""
    if not USE_IMAGE_MODEL or image_model is None:
        return None
    try:
        img_t = image_preprocess(pil_img).unsqueeze(0)  # [1,3,224,224]
        with torch.no_grad():
            feats = image_model(img_t)  # [1, C, H, W]
            emb = feats.mean(dim=[2,3])  # [1, C]
            emb_np = emb.cpu().numpy().ravel()
            return emb_np
    except Exception:
        return None

def process_attachment(path_str):
    """
    Returns: (attachment_text, attachment_numeric_features_dict)
    numeric features: attachment_text_len (words), num_images, avg_image_emb_norm
    """
    if not path_str or not isinstance(path_str, str) or not path_str.strip():
        return "", {"attachment_text_len": 0.0, "num_images": 0.0, "avg_image_emb_norm": 0.0}

    p = Path(path_str)
    if not p.exists():
        # try relative to current dir or as given; if not exists, return zeros
        return "", {"attachment_text_len": 0.0, "num_images": 0.0, "avg_image_emb_norm": 0.0}

    suffix = p.suffix.lower()
    att_text = ""
    image_embeddings = []

    # text-like files
    if suffix in {".txt", ".log", ".csv"}:
        att_text = extract_text_from_txt(p)
    elif suffix in {".pdf"}:
        # extract text and images
        att_text = extract_text_from_pdf(p)
        imgs = extract_images_from_pdf(p)
        for im in imgs:
            emb = compute_image_embedding(im)
            if emb is not None:
                image_embeddings.append(emb)
            # also optionally OCR images if pytesseract available
            if USE_PYTESSERACT:
                try:
                    att_text += "\n" + (pytesseract.image_to_string(im) or "")
                except Exception:
                    pass
    elif suffix in {".docx", ".doc"}:
        if USE_DOCX and suffix == ".docx":
            att_text = extract_text_from_docx(p)
        else:
            # fallback: try read as text
            att_text = extract_text_from_txt(p)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}:
        # single image file attachment
        if USE_PYTORCH:
            try:
                im = Image.open(str(p)).convert("RGB")
                emb = compute_image_embedding(im)
                if emb is not None:
                    image_embeddings.append(emb)
                if USE_PYTESSERACT:
                    att_text += "\n" + (pytesseract.image_to_string(im) or "")
            except Exception:
                pass
    else:
        # unknown extension: try PDF text, then raw read
        att_text = extract_text_from_pdf(p) or extract_text_from_txt(p)

    # build numeric features
    words = att_text.split()
    attachment_text_len = float(len(words))
    num_images = float(len(image_embeddings))
    avg_image_emb_norm = 0.0
    if image_embeddings:
        norms = [float(np.linalg.norm(e)) for e in image_embeddings]
        avg_image_emb_norm = float(np.mean(norms))

    return att_text, {
        "attachment_text_len": attachment_text_len,
        "num_images": num_images,
        "avg_image_emb_norm": avg_image_emb_norm
    }

# ---------------- LOAD & PREP DATA ----------------
if not Path(DATA_CSV).exists():
    raise SystemExit(f"Dataset CSV not found: {DATA_CSV}")

df = pd.read_csv(DATA_CSV)

# combine subject + body; keep original columns
df['subject'] = df.get('subject', pd.Series([""]*len(df)))
df['body'] = df.get('body', pd.Series([""]*len(df)))
df['text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

# If CSV has an 'attachment_path' column, we will process attachments; otherwise we'll skip gracefully.
attachment_paths = df.get('attachment_path', None)

# Base numeric features: existing ones you had
base_numeric_features = ['has_links', 'num_links', 'has_attachment', 'phishing_score']

# new attachment-derived numeric features we will add
attachment_numeric_names = ['attachment_text_len', 'num_images', 'avg_image_emb_norm']

# ensure numeric base columns exist (fill with zeros if missing)
for col in base_numeric_features:
    if col not in df.columns:
        df[col] = 0

# Process attachments (if present) and append attachment text into main text
attachment_texts = []
attachment_numeric_rows = []
if attachment_paths is not None:
    print("Processing attachments (this may take a while)...")
    for idx, ap in enumerate(attachment_paths.fillna("")):
        att_text, att_nums = process_attachment(str(ap))
        attachment_texts.append(att_text)
        attachment_numeric_rows.append([att_nums[n] for n in attachment_numeric_names])
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1}/{len(attachment_paths)} attachments...")
else:
    # no attachment column — create zeros
    attachment_texts = [""] * len(df)
    attachment_numeric_rows = [[0.0, 0.0, 0.0] for _ in range(len(df))]

# Append attachment text to email text for TF-IDF
df['full_text'] = (df['text'].fillna("") + " " + pd.Series(attachment_texts).fillna("")).astype(str)

# Build numeric feature matrix: base numeric + attachment numeric
attachment_numeric_arr = np.array(attachment_numeric_rows, dtype=float)
base_numeric_arr = df[base_numeric_features].fillna(0).astype(float).values
X_numeric_all = np.hstack([base_numeric_arr, attachment_numeric_arr])

# ---------------- PREPROCESSING ----------------
# TF-IDF for text (subject+body+attachment_text)
tfidf = TfidfVectorizer(max_features=8000, stop_words='english')
X_text = tfidf.fit_transform(df['full_text'])

# Standardize numeric features
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_numeric_all)
X_num_sparse = csr_matrix(X_num_scaled)

# Combine features
X = hstack([X_text, X_num_sparse])

# ---------------- LABELS ----------------
# Expect df['label'] to be present with values like 'phishing'/'legitimate' or 1/0
if 'label' not in df.columns:
    raise SystemExit("CSV missing 'label' column (required).")

# normalize label strings to 'phishing' or 'legitimate' (if numeric, keep as is)
y_raw = df['label'].astype(str).str.lower().values
# map common forms to 'phishing'/'legitimate'
def map_label(x):
    if x in {'1','true','phishing','phish','malicious'}:
        return 'phishing'
    if x in {'0','false','legitimate','ham','benign','not phishing'}:
        return 'legitimate'
    return x
y_mapped = np.array([map_label(v) for v in y_raw])

# filter unknown labels if any
mask_known = ~np.isin(y_mapped, ['unknown', ''])
X_known = X[mask_known]
y_known = y_mapped[mask_known]

if X_known.shape[0] == 0:
    raise SystemExit("No labeled examples after filtering unknown labels.")

# ---------------- SPLIT & TRAIN ----------------
X_train, X_test, y_train, y_test = train_test_split(X_known, y_known, test_size=0.2, random_state=42, stratify=y_known)

model = LinearSVC(max_iter=20000)
print("Training LinearSVC on combined text + numeric features...")
model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
print("----- Classification Report -----")
print(classification_report(y_test, y_pred))

# ---------------- SAVE ARTIFACTS ----------------
joblib.dump(model, MODEL_FILE)
joblib.dump(tfidf, VECTORIZER_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"✅ Model saved as {MODEL_FILE}")
print(f"✅ TF-IDF vectorizer saved as {VECTORIZER_FILE}")
print(f"✅ Numeric scaler saved as {SCALER_FILE}")

# ---------------- PREDICTION FUNCTION ----------------
def predict_email(subject, body, has_links=0, num_links=0, has_attachment=0, phishing_score=0.0, attachment_path=""):
    """
    Load saved artifacts and run a single prediction. Returns label string.
    If optional attachment_path provided, it will be processed similarly to training.
    """
    mdl = joblib.load(MODEL_FILE)
    vec = joblib.load(VECTORIZER_FILE)
    sc = joblib.load(SCALER_FILE)

    # Build text
    subject = "" if subject is None else str(subject)
    body = "" if body is None else str(body)
    att_text, att_nums = process_attachment(str(attachment_path)) if attachment_path else ("", {"attachment_text_len":0.0,"num_images":0.0,"avg_image_emb_norm":0.0})

    full_text = f"{subject} {body} {att_text}".strip()
    X_text_vect = vec.transform([full_text])

    # Build numeric vector in same order as training: base_numeric_features then attachment_numeric_names
    numeric_input = [float(has_links), float(num_links), float(has_attachment), float(phishing_score),
                     float(att_nums["attachment_text_len"]), float(att_nums["num_images"]), float(att_nums["avg_image_emb_norm"])]
    X_num_scaled = sc.transform([numeric_input])
    X_in = hstack([X_text_vect, csr_matrix(X_num_scaled)])

    pred = mdl.predict(X_in)[0]
    # if you want probabilities, note LinearSVC doesn't provide predict_proba. You could use CalibratedClassifierCV for that.
    return pred

# ---------------- QUICK TEST ----------------
if __name__ == "__main__":
    s = "Security Alert: Verify Your Account"
    b = "We detected unusual activity. Click here to reset your password immediately."
    # optionally specify attachment_path if you have a sample file on disk
    label = predict_email(s, b, has_links=1, num_links=1, has_attachment=0, phishing_score=0.8, attachment_path="")
    print("Predicted label:", label)
