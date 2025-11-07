#!/usr/bin/env python3
"""
Train a simple phishing detector from .msg / .eml / plain-text email files.

Features:
 - Accepts either a single file or a directory (searched recursively).
 - Finds files in nested subfolders (os.walk).
 - Supports extensions: .msg (Outlook), .eml (RFC-822), and plain-text (.txt/.mail).
 - Labels by folder name (if folder contains 'fake', 'phish', 'spam' -> phishing;
   'real', 'ham' -> legitimate). Falls back to a simple keyword-based heuristic.
 - TF-IDF vectorizer + LogisticRegression classifier. Saves model and vectorizer to disk.
 - Optional: use --exts to list allowed extensions (comma-separated, with or without dots).
"""

import os
import re
import argparse
import sys
import pandas as pd
import email
from email import policy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import nltk

# Optional: extract_msg is used for .msg files. If not available, .msg files will be skipped.
try:
    import extract_msg
    HAS_EXTRACT_MSG = True
except Exception:
    HAS_EXTRACT_MSG = False

# ---- Download stopwords (quiet) ----
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# ---- Helper: clean text ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# ---- Helper: read plain-text "email-like" files (.txt / .mail) ----
def read_plain_text_email(path):
    """
    Read a plain-text file that may contain RFC-like headers and body.
    Heuristic:
      - Try to decode as utf-8, fallback to latin-1 with errors ignored.
      - Look for a line starting with 'Subject:' within the first 100 lines.
      - Everything after the first blank line is treated as body (RFC-822 heuristic).
    Returns dict {"subject":..., "body":..., "path":...} or None
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            text = raw.decode("latin-1", errors="ignore")

        # split into lines and try to find Subject: header
        lines = text.splitlines()
        subject = ""
        for ln in lines[:100]:
            if ln.lower().startswith("subject:"):
                subject = ln.split(":", 1)[1].strip()
                break

        # find first blank line -> headers/body split
        body = ""
        if "" in lines:
            idx = lines.index("")
            body = "\n".join(lines[idx+1:]).strip()
        else:
            # no header/body split found; treat entire text as body
            body = text.strip()

        return {"subject": subject, "body": body, "path": path}
    except Exception as e:
        print(f"⚠️  Error reading plain text file '{path}': {e}")
        return None

# ---- Helper: read a single .msg or .eml or .txt file ----
def read_email_file(path, supported_exts=(".msg", ".eml", ".txt", ".mail")):
    """
    Read .msg (Outlook) via extract_msg if available, .eml (RFC-822) via email module,
    and fallback to plain-text (.txt/.mail) via read_plain_text_email.
    Returns dict {"subject":..., "body":..., "path":...} or None on failure.
    """
    try:
        lname = path.lower()
        if lname.endswith(".msg"):
            if not HAS_EXTRACT_MSG:
                print(f"⚠️  Skipping .msg file (extract_msg not installed): {path}")
                return None
            try:
                msg = extract_msg.Message(path)
                subject = msg.subject or ""
                body = msg.body or ""
                return {"subject": subject, "body": body, "path": path}
            except Exception as e:
                print(f"⚠️  extract_msg failed for '{path}': {e}")
                return None

        elif lname.endswith(".eml"):
            # read RFC-822 .eml safely
            try:
                with open(path, "rb") as fh:
                    em = email.message_from_binary_file(fh, policy=policy.default)
                subject = em.get("subject", "") or ""
                # get plain text body (join parts if multipart)
                body = ""
                if em.is_multipart():
                    for part in em.walk():
                        ctype = part.get_content_type()
                        disp = str(part.get("Content-Disposition", ""))
                        if ctype == "text/plain" and "attachment" not in disp:
                            try:
                                body_piece = part.get_content()
                            except Exception:
                                body_piece = part.get_payload(decode=True)
                                if isinstance(body_piece, bytes):
                                    body_piece = body_piece.decode(errors="ignore")
                            if body_piece:
                                body += body_piece + "\n"
                else:
                    try:
                        body = em.get_content()
                    except Exception:
                        payload = em.get_payload(decode=True)
                        if isinstance(payload, bytes):
                            body = payload.decode(errors="ignore")
                        else:
                            body = str(payload)
                return {"subject": subject, "body": body, "path": path}
            except Exception as e:
                print(f"⚠️  Error parsing .eml '{path}': {e}")
                return None

        elif lname.endswith((".txt", ".mail")):
            return read_plain_text_email(path)

        else:
            # unsupported extension
            return None
    except Exception as e:
        print(f"⚠️  Error reading '{path}': {e}")
        return None

# ---- Helper: collect emails (recursive) ----
def collect_emails_recursive(path, supported_exts=(".msg", ".eml", ".txt", ".mail")):
    """
    If path is a file -> read it (if supported).
    If path is a dir -> walk recursively and read every supported file found.
    Returns list of dicts: {"subject":..., "body":..., "path":...}
    """
    emails = []

    supported_exts = tuple(e if e.startswith(".") else "." + e for e in supported_exts)

    if os.path.isfile(path):
        lname = path.lower()
        if lname.endswith(supported_exts):
            print(f"📄 Reading single file: {path}")
            item = read_email_file(path, supported_exts)
            if item:
                emails.append(item)
        else:
            print(f"❌ Provided file is not a supported email type: {path}")
        return emails

    if os.path.isdir(path):
        print(f"📂 Recursively reading directory: {path}")
        for root, dirs, files in os.walk(path):
            for fname in sorted(files):
                if fname.lower().endswith(supported_exts):
                    fpath = os.path.join(root, fname)
                    item = read_email_file(fpath, supported_exts)
                    if item:
                        emails.append(item)
        return emails

    print(f"❌ Path not found: {path}")
    return emails

# ---- Helper: label from folder name if possible ----
def label_from_path(path):
    """
    Inspect path string and return:
      1 -> phishing (fake)
      0 -> legitimate (real)
      None -> unknown
    """
    p = path.replace("\\", "/").lower()
    # phishing indicators
    for token in ("/fake/", "/phish/", "/phishing/", "/spam/", "/scam/"):
        if token in p:
            return 1
    # legitimate indicators
    for token in ("/real/", "/ham/", "/legit/"):
        if token in p:
            return 0
    return None

def parse_exts_arg(exts_arg):
    """
    Parse a comma-separated extensions list into a tuple like ('.eml', '.msg', '.txt')
    Accepts values like 'eml,msg' or '.eml,.msg'.
    """
    parts = [p.strip() for p in exts_arg.split(",") if p.strip()]
    normalized = []
    for p in parts:
        if not p.startswith("."):
            p = "." + p
        normalized.append(p.lower())
    return tuple(normalized)

def main():
    parser = argparse.ArgumentParser(description="Train phishing email detector from files (recursive).")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Folder (searched recursively) or single file. If relative, resolved against script base dir.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Where to save the trained model (default: <base>/phishing_model.pkl)")
    parser.add_argument("--vectorizer-path", type=str, default=None,
                        help="Where to save the vectorizer (default: <base>/vectorizer.pkl)")
    parser.add_argument("--max-features", type=int, default=5000, help="Tfidf max_features")
    parser.add_argument("--prefer-dir-labels", action="store_true",
                        help="If set, prefer directory-based labels when available (default behavior uses dir labels when present).")
    parser.add_argument("--exts", type=str, default=".msg,.eml,.txt,.mail",
                        help="Comma-separated list of accepted extensions (e.g. '.eml,.msg,.txt').")
    args = parser.parse_args()

    # base dir: parent of this script (keeps same behavior)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # resolve dataset path
    if os.path.isabs(args.data_dir):
        dataset_path = os.path.normpath(args.data_dir)
    else:
        dataset_path = os.path.normpath(os.path.join(base_dir, args.data_dir))

    model_path = args.model_path if args.model_path else os.path.join(base_dir, "phishing_model.pkl")
    vectorizer_path = args.vectorizer_path if args.vectorizer_path else os.path.join(base_dir, "vectorizer.pkl")

    supported_exts = parse_exts_arg(args.exts)

    print("Base dir      :", base_dir)
    print("Dataset path  :", dataset_path)
    print("Model will be :", model_path)
    print("Vectorizer will be :", vectorizer_path)
    print("Accepted file extensions:", supported_exts)
    if not HAS_EXTRACT_MSG and any(e == ".msg" for e in supported_exts):
        print("⚠️  Note: 'extract_msg' not installed. .msg files will be skipped. Install with: pip install extract_msg")

    # collect emails recursively
    emails = collect_emails_recursive(dataset_path, supported_exts)
    if not emails:
        print("❌ No emails loaded. Make sure the path exists and contains supported files (including nested subfolders).")
        sys.exit(1)

    df = pd.DataFrame(emails)
    print(f"✅ Loaded {len(df)} email(s) from files")

    # Label by directory when possible
    df["label_dir"] = df["path"].apply(label_from_path)

    # --- Label emails (directory-based preferred, fallback to keyword-based) ---
    # Keyword heuristic
    keyword_re = r"claim|free|update|verify|urgent|alert|password|win|invoice|payment|suspended"
    keyword_label = df["subject"].fillna("").str.contains(keyword_re, case=False, na=False).astype(int)

    # Combine: prefer dir label when available, else use keyword label
    if args.prefer_dir_labels:
        df["is_phishing"] = df["label_dir"].where(df["label_dir"].notnull(), keyword_label).astype(int)
    else:
        # existing behavior: directory labels used when present (same as prefer_dir_labels)
        df["is_phishing"] = df["label_dir"].where(df["label_dir"].notnull(), keyword_label).astype(int)

    # Inform about labeling balance and how many used directory labels
    n_dir_labels = df["label_dir"].notnull().sum()
    print(f"ℹ️  Using directory-based labels for {n_dir_labels} files (if folder names contained 'fake'/'real' etc.)")
    print(f"ℹ️  Label distribution (is_phishing):\n{df['is_phishing'].value_counts(dropna=False)}")

    # --- Clean text ---
    df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)

    # --- Split data ---
    # If only one class present, still proceed but will warn and skip stratify
    if df["is_phishing"].nunique() == 1:
        print("⚠️  Warning: only one class present in labels. Model will train but evaluation will be trivial.")
        strat = None
    else:
        strat = df["is_phishing"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["is_phishing"], test_size=0.2, random_state=42, stratify=strat
        )
    except Exception as e:
        print("⚠️  train_test_split with stratify failed, falling back to plain split:", e)
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"], df["is_phishing"], test_size=0.2, random_state=42
        )

    # --- Vectorize ---
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Train model ---
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_vec)
    print("\n✅ Training complete!")
    try:
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print("\nReport:\n", classification_report(y_test, y_pred))
    except Exception as e:
        print("⚠️  Could not compute evaluation metrics:", e)

    # --- Save model & vectorizer ---
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"\n💾 Saved model → {model_path}")
    print(f"💾 Saved vectorizer → {vectorizer_path}")

if __name__ == "__main__":
    main()
