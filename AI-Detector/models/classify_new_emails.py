import os
import re
import email
from bs4 import BeautifulSoup
import joblib
from scipy.sparse import hstack

# ---------------- CONFIG ----------------
EMAIL_DIR = "new_emails"   # Folder with new .eml files
MODEL_FILE = "phishing_email_model.pkl"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SCALER_FILE = "numeric_scaler.pkl"

# Suspicious domains (for phishing_score computation)
PHISHING_DOMAINS = ["secure-paypal.com", "login-update.net", "verify-account.org"]

# Suspicious keywords
SUSPICIOUS_KEYWORDS = [
    "verify your account", "update password", "urgent", "security alert",
    "click here", "reset password", "confirm identity", "unusual activity",
    "login attempt", "invoice attached", "suspended", "payment required"
]

# ---------------- LOAD MODEL & PREPROCESSORS ----------------
model = joblib.load(MODEL_FILE)
tfidf = joblib.load(VECTORIZER_FILE)
scaler = joblib.load(SCALER_FILE)

# ---------------- HELPER FUNCTIONS ----------------
def get_domain(email_address):
    match = re.search(r"@([A-Za-z0-9.-]+)", email_address)
    return match.group(1).lower() if match else ""

def contains_suspicious_keywords(text):
    text = text.lower()
    if not text:
        return 0
    count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text)
    return min(1.0, count / len(SUSPICIOUS_KEYWORDS))

def compute_phishing_score(domain, subject, body, links):
    # Domain risk
    if any(bad in domain for bad in PHISHING_DOMAINS):
        domain_score = 1.0
    else:
        domain_score = 0.5  # Unknown domain
    # Keyword risk
    keyword_score = max(contains_suspicious_keywords(subject), contains_suspicious_keywords(body))
    # Link risk
    link_score = 0.0
    for link in links:
        for bad in PHISHING_DOMAINS:
            if bad in link:
                link_score = 1.0
                break
    # Weighted average
    score = 0.4 * domain_score + 0.4 * keyword_score + 0.2 * link_score
    return round(score, 2)

def parse_eml(file_path):
    """Extract sender, subject, body, links, attachments from .eml"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        msg = email.message_from_file(f)

    sender = msg.get("From", "")
    subject = msg.get("Subject", "")
    domain = get_domain(sender)

    body = ""
    has_attachment = 0
    attachment_type = "none"
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                body += part.get_payload(decode=True).decode(errors="ignore") + " "
            except:
                pass
        elif part.get_content_type() == "text/html":
            try:
                html = part.get_payload(decode=True).decode(errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                body += soup.get_text() + " "
            except:
                pass
        if part.get_content_disposition() == "attachment":
            has_attachment = 1
            attachment_type = part.get_content_type().split("/")[-1]

    # Links
    links = re.findall(r"http[s]?://\S+", body)
    has_links = 1 if links else 0
    num_links = len(links)

    phishing_score = compute_phishing_score(domain, subject, body, links)

    return {
        "sender": sender,
        "subject": subject,
        "body": body.strip(),
        "has_links": has_links,
        "num_links": num_links,
        "has_attachment": has_attachment,
        "attachment_type": attachment_type,
        "phishing_score": phishing_score
    }

def classify_email(parsed_email):
    """Predict phishing / benign label"""
    text_vect = tfidf.transform([parsed_email['subject'] + ' ' + parsed_email['body']])
    numeric_scaled = scaler.transform([[parsed_email['has_links'], parsed_email['num_links'],
                                        parsed_email['has_attachment'], parsed_email['phishing_score']]])
    X_input = hstack([text_vect, numeric_scaled])
    prediction = model.predict(X_input)
    return prediction[0]

# ---------------- PROCESS ALL EMAILS ----------------
for file in os.listdir(EMAIL_DIR):
    if file.endswith(".eml"):
        path = os.path.join(EMAIL_DIR, file)
        parsed = parse_eml(path)
        label = classify_email(parsed)
        print(f"File: {file}")
        print(f"Sender: {parsed['sender']}")
        print(f"Subject: {parsed['subject']}")
        print(f"Predicted label: {label}")
        print(f"Phishing score: {parsed['phishing_score']}")
        print("-" * 50)
