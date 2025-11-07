import os
import extract_msg

# Folders
INPUT_FOLDER = "../emails_msg"
OUTPUT_FOLDER = "../emails_eml"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def convert_msg_to_eml(msg_path, eml_path):
    try:
        msg = extract_msg.Message(msg_path)

        # Build simple EML content
        eml_content = f"""From: {msg.sender or ''}
To: {msg.to or ''}
Cc: {msg.cc or ''}
Subject: {msg.subject or ''}
Date: {msg.date or ''}

{msg.body or ''}
"""

        # Save EML file
        with open(eml_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(eml_content)

        print(f"✅ Converted: {os.path.basename(msg_path)}")

    except Exception as e:
        print(f"❌ Error converting {msg_path}: {e}")

def main():
    for file in os.listdir(INPUT_FOLDER):
        if file.lower().endswith(".msg"):
            msg_path = os.path.join(INPUT_FOLDER, file)
            eml_path = os.path.join(OUTPUT_FOLDER, os.path.splitext(file)[0] + ".eml")
            convert_msg_to_eml(msg_path, eml_path)

if __name__ == "__main__":
    main()
