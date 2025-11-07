import os
import csv
from email import policy
from email.parser import BytesParser

INPUT_FOLDER = "../emails_eml"
OUTPUT_CSV = "../emails_dataset.csv"

def parse_eml(file_path):
    """Extract subject, sender, recipient, and body text from an .eml file."""
    with open(file_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)

    subject = msg["subject"] or ""
    sender = msg["from"] or ""
    recipient = msg["to"] or ""
    body = ""

    # Extract plain text body
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body = msg.get_content()

    return {
        "subject": subject.strip(),
        "sender": sender.strip(),
        "recipient": recipient.strip(),
        "body": body.strip(),
    }

def main():
    emails_data = []
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".eml"):
            path = os.path.join(INPUT_FOLDER, file)
            try:
                email_data = parse_eml(path)
                emails_data.append(email_data)
                print(f"✅ Parsed: {file}")
            except Exception as e:
                print(f"❌ Error parsing {file}: {e}")

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "sender", "recipient", "body"])
        writer.writeheader()
        writer.writerows(emails_data)

    print(f"\n📄 Dataset saved to {OUTPUT_CSV} ({len(emails_data)} emails).")

if __name__ == "__main__":
    main()
