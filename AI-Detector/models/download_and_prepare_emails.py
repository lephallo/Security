#!/usr/bin/env python3
"""
downloads public corpora (SpamAssassin + Enron fallback), extracts, and
selects 20 spam (fake) and 20 ham (real) messages into:
  emails_msg/emails_msg/fake/
  emails_msg/emails_msg/real/

Sources:
- SpamAssassin public corpus (spam + easy_ham). (Apache)
- Enron dataset (CMU / SNAP mirror) used as fallback for ham.
"""

import os
import sys
import shutil
import tarfile
import tempfile
import random
import pathlib
from urllib.parse import urlsplit
from urllib.request import urlretrieve

# --- Config ---
OUT_BASE = pathlib.Path("emails_msg") / "emails_msg"
OUT_FAKE = OUT_BASE / "fake"
OUT_REAL = OUT_BASE / "real"
N_FAKE = 20
N_REAL = 20

# Public dataset URLs (direct)
SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SPAM_URL = SPAMASSASSIN_BASE + "20021010_spam.tar.bz2"
EASY_HAM_URL = SPAMASSASSIN_BASE + "20030228_easy_ham.tar.bz2"

# Enron fallback (if we need more ham)
ENRON_URL = "http://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

# helper: safe extraction to avoid path traversal
def safe_extract(tar, path=".", members=None):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not os.path.commonpath([os.path.abspath(path)]) == os.path.commonpath([os.path.abspath(path), os.path.abspath(member_path)]):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members)

def ensure_dirs():
    OUT_FAKE.mkdir(parents=True, exist_ok=True)
    OUT_REAL.mkdir(parents=True, exist_ok=True)

def download(url, dst_folder):
    print(f"Downloading {url} ...")
    dst_folder = pathlib.Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(urlsplit(url).path)
    out_path = dst_folder / filename
    if out_path.exists():
        print(" - already downloaded:", out_path)
        return str(out_path)
    urlretrieve(url, str(out_path))
    print(" - saved to:", out_path)
    return str(out_path)

def collect_message_files(root_dir):
    """
    Walks root_dir and returns a list of file paths that look like email messages.
    We treat files with no extension or common mail-like extensions as candidates.
    """
    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            # skip tiny files
            try:
                if os.path.getsize(fp) < 50:
                    continue
            except OSError:
                continue
            # common mail containers: mbox (no extension), .eml, or plain files
            candidates.append(fp)
    return candidates

def pick_and_copy(candidates, dest_dir, n, prefix):
    random.shuffle(candidates)
    picked = candidates[:n]
    for i, src in enumerate(picked):
        ext = ".eml" if src.lower().endswith(".eml") else ".txt"
        dest = dest_dir / f"{prefix}_{i+1:02d}{ext}"
        shutil.copy2(src, dest)
    return len(picked)

def extract_and_gather(tar_path, tmpdir):
    gathered = []
    try:
        if tarfile.is_tarfile(tar_path):
            with tarfile.open(tar_path, "r:*") as tar:
                # extract into tmpdir/subfolder
                sub = pathlib.Path(tmpdir) / (pathlib.Path(tar_path).stem)
                sub.mkdir(parents=True, exist_ok=True)
                safe_extract(tar, str(sub))
                gathered.extend(collect_message_files(sub))
    except Exception as e:
        print("Error extracting", tar_path, ":", e)
    return gathered

def main():
    ensure_dirs()
    with tempfile.TemporaryDirectory() as tmp:
        all_spam_candidates = []
        all_ham_candidates = []

        # 1) download spamassassin spam + easy_ham
        try:
            spam_tar = download(SPAM_URL, tmp)
            easy_ham_tar = download(EASY_HAM_URL, tmp)
        except Exception as e:
            print("Download failed:", e)
            sys.exit(1)

        # 2) extract and gather
        print("Extracting spam archive...")
        all_spam_candidates += extract_and_gather(spam_tar, tmp)
        print("Found spam candidate count:", len(all_spam_candidates))

        print("Extracting easy_ham archive...")
        all_ham_candidates += extract_and_gather(easy_ham_tar, tmp)
        print("Found easy_ham candidate count:", len(all_ham_candidates))

        # 3) if ham not enough, download Enron and extract a bit
        if len(all_ham_candidates) < N_REAL:
            print("Not enough ham from SpamAssassin; downloading Enron (may be large)...")
            enron_tar = download(ENRON_URL, tmp)
            enron_msgs = extract_and_gather(enron_tar, tmp)
            print("Enron candidates:", len(enron_msgs))
            all_ham_candidates += enron_msgs

        # 4) pick and copy
        if len(all_spam_candidates) < N_FAKE:
            print(f"Warning: only {len(all_spam_candidates)} spam candidates found (need {N_FAKE}).")
        if len(all_ham_candidates) < N_REAL:
            print(f"Warning: only {len(all_ham_candidates)} ham candidates found (need {N_REAL}).")

        n_fake = min(N_FAKE, len(all_spam_candidates))
        n_real = min(N_REAL, len(all_ham_candidates))

        print(f"Copying {n_fake} spam -> {OUT_FAKE}")
        pick_and_copy(all_spam_candidates, OUT_FAKE, n_fake, "phish")
        print(f"Copying {n_real} ham -> {OUT_REAL}")
        pick_and_copy(all_ham_candidates, OUT_REAL, n_real, "real")

        print("Done. You should now have:")
        print(" -", len(list(OUT_FAKE.glob("*"))), "fake files in", OUT_FAKE)
        print(" -", len(list(OUT_REAL.glob("*"))), "real files in", OUT_REAL)

if __name__ == "__main__":
    main()
