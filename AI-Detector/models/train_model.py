import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords

# ---------------------- Setup ----------------------
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

# ---------------------- Clean text function ----------------------
def clean_text(text):
    """Lowercase, remove special characters, and filter stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# ---------------------- Load dataset ----------------------
# Make sure you have your CSV file at: AI_Phishing_Project/emails_dataset.csv
DATA_CSV = os.path.join(os.path.dirname(__file__), "../emails_dataset.csv")
df = pd.read_csv(DATA_CSV)

# Ensure necessary columns exist
if "subject" not in df.columns or "body" not in df.columns:
    raise ValueError("CSV must have 'subject' and 'body' columns.")

# Add label column automatically (you can replace with your own labels)
df["is_phishing"] = df["subject"].str.contains(
    r"claim|off|free|verify|password|urgent|account", case=False, na=False
).astype(int)

# Combine and clean text
df["text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).apply(clean_text)

# ---------------------- Split dataset ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["is_phishing"], test_size=0.2, random_state=42
)

# ---------------------- Vectorize text ----------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------- Train model ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------------------- Evaluate ----------------------
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------- Save model and vectorizer ----------------------
save_dir = os.path.dirname(__file__)
joblib.dump(model, os.path.join(save_dir, "phishing_model.pkl"))
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))

print("\n🎉 Model and vectorizer saved successfully in Scripts folder!")
