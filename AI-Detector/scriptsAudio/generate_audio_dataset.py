"""
generate_audio_dataset.py

- Creates small real + fake audio samples.
- Converts them to mel-spectrogram PNGs.
- Robust to Windows torchaudio backend problems (uses soundfile backend or librosa fallback).
- Tries pyttsx3 for "real" voices and falls back to gTTS if needed.

Requirements (install in venv):
pip install gTTS pyttsx3 soundfile torchaudio matplotlib librosa

On Windows you may also need:
pip install pypiwin32
"""

import os
from pathlib import Path
import traceback

# Audio / TTS / Processing libs
try:
    # torchaudio preferred for loading and transforms
    import torchaudio
    import torchaudio.transforms as TTA
except Exception:
    torchaudio = None

# We'll use librosa as a robust fallback for loading audio
try:
    import librosa
    import librosa.display
except Exception:
    librosa = None

import matplotlib.pyplot as plt

# TTS engines
from gtts import gTTS
try:
    import pyttsx3
except Exception:
    pyttsx3 = None

# ----------------------------
# 1️⃣ Directory setup
# ----------------------------
real_audio_dir = Path("datasets/audio_samples/real")
fake_audio_dir = Path("datasets/audio_samples/fake")
real_spec_dir = Path("datasets/audio_spectrograms/real")
fake_spec_dir = Path("datasets/audio_spectrograms/fake")

for d in [real_audio_dir, fake_audio_dir, real_spec_dir, fake_spec_dir]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------
# 2️⃣ Sentences for speech generation
# ----------------------------
sentences = [
    "Artificial intelligence is transforming cybersecurity.",
    "Deepfake voices can be used to imitate anyone.",
    "Machine learning models can detect phishing emails.",
    "This is a sample real audio for testing.",
    "Protect your identity online using two factor authentication.",
    "Cyber attacks are increasing every year.",
    "Email attachments can contain phishing links.",
    "Deep learning helps identify fake media content.",
    "Audio deepfakes are hard to detect by humans.",
    "Always verify suspicious messages before opening them."
]

# ----------------------------
# 3️⃣ Generate 'real' audio with pyttsx3 (preferred), fallback to gTTS
# ----------------------------
def generate_real_with_pyttsx3(target_dir: Path, texts):
    if pyttsx3 is None:
        raise RuntimeError("pyttsx3 is not installed or failed to import.")
    engine = pyttsx3.init()
    # Optionally set voice properties here (volume, rate)
    try:
        for i, text in enumerate(texts):
            file_path = target_dir / f"real_{i+1:02d}.wav"
            engine.save_to_file(text, str(file_path))
        engine.runAndWait()
    finally:
        # some engines need to be stopped/cleaned - safe guard
        try:
            engine.stop()
        except Exception:
            pass

def generate_with_gtts(target_dir: Path, texts, prefix="fake"):
    for i, text in enumerate(texts):
        tts = gTTS(text)
        file_path = target_dir / f"{prefix}_{i+1:02d}.wav"
        tts.save(str(file_path))

# Try pyttsx3 first for 'real' voices; if it fails, fallback to gTTS for both real+fake
real_generated = False
try:
    if pyttsx3 is not None:
        print("Attempting to generate 'real' audio with pyttsx3 (offline)...")
        generate_real_with_pyttsx3(real_audio_dir, sentences)
        real_generated = True
        print(f"✅ Saved {len(sentences)} real audio files (pyttsx3).")
    else:
        raise RuntimeError("pyttsx3 unavailable")
except Exception as e:
    print("⚠️ pyttsx3 failed (or not available). Falling back to gTTS for 'real' audio.")
    print("Reason:", e)
    try:
        generate_with_gtts(real_audio_dir, sentences, prefix="real")
        real_generated = True
        print(f"✅ Saved {len(sentences)} real audio files (gTTS fallback).")
    except Exception as e2:
        print("❌ Failed to create real audio with gTTS fallback.")
        traceback.print_exc()

# Generate 'fake' audio with gTTS
try:
    print("Generating 'fake' audio with gTTS...")
    generate_with_gtts(fake_audio_dir, sentences, prefix="fake")
    print(f"✅ Saved {len(sentences)} fake audio files (gTTS).")
except Exception:
    print("❌ Failed to create fake audio with gTTS. See error:")
    traceback.print_exc()

# ----------------------------
# 4️⃣ Create spectrograms
# ----------------------------

# Utility: try to load with torchaudio (soundfile backend) else fallback to librosa
def load_audio(audio_path):
    audio_path_str = str(audio_path)
    # Try torchaudio first (explicit backend soundfile)
    if torchaudio is not None:
        try:
            # ensure soundfile backend is available; this will raise if not
            waveform, sr = torchaudio.load(audio_path_str, backend="soundfile")
            return waveform, int(sr)
        except Exception as e:
            # Continue to fallback
            print(f"torchaudio.load(soundfile) failed for {audio_path.name}: {e}")

    # Fallback to librosa
    if librosa is not None:
        try:
            y, sr = librosa.load(audio_path_str, sr=None)  # keep original sr
            # librosa returns 1d array (mono); convert to torch-like shape (1, n)
            import numpy as np
            waveform = np.asfortranarray(y)
            # shape to (1, n) and convert to torch tensor if torchaudio exists
            if torchaudio is not None:
                import torch
                return torch.from_numpy(waveform).unsqueeze(0), int(sr)
            else:
                return waveform, int(sr)
        except Exception as e:
            print(f"librosa.load failed for {audio_path.name}: {e}")

    # Last resort: raise
    raise RuntimeError(f"Could not load audio file {audio_path_str} with torchaudio or librosa.")

# Create spectrogram using torchaudio transforms if available, else use librosa for spectrogram
def create_spectrogram(audio_path: Path, save_path: Path):
    try:
        loaded = load_audio(audio_path)
    except Exception as e:
        print(f"Failed to load {audio_path}: {e}")
        return False

    # If torchaudio is available and we got torch tensors, use torchaudio transforms
    if torchaudio is not None and hasattr(loaded[0], "dtype"):
        waveform, sr = loaded
        # waveform expected shape: (channels, samples)
        if waveform.ndim == 1:
            # convert to (1, n)
            import torch
            waveform = waveform.unsqueeze(0)
        # Convert to mono by averaging channels if multiple
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # create mel spectrogram
        try:
            mel = TTA.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
            db = TTA.AmplitudeToDB()(mel)
            # db shape: (channel=1, n_mels, time)
            # Convert to numpy for matplotlib
            img = db[0].cpu().numpy()
        except Exception as e:
            print(f"torchaudio transforms failed for {audio_path.name}: {e}")
            print("Attempting librosa fallback to compute spectrogram.")
            img = None
    else:
        img = None

    # If img is None, compute with librosa (if available)
    if img is None:
        if librosa is None:
            print("No method available to compute spectrogram (librosa not installed).")
            return False
        try:
            import numpy as np
            y, sr = librosa.load(str(audio_path), sr=None)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = S_db
        except Exception as e:
            print(f"Failed to compute spectrogram with librosa for {audio_path.name}: {e}")
            return False

    # Plot and save without axes / whitespace
    plt.figure(figsize=(4, 4), dpi=100)
    plt.axis("off")
    plt.imshow(img, aspect="auto", origin="lower")
    plt.tight_layout(pad=0)
    try:
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
    except Exception as e:
        print(f"Failed to save spectrogram {save_path}: {e}")
        plt.close()
        return False
    plt.close()
    return True

# Process real files
real_files = sorted(real_audio_dir.glob("*.wav"))
fake_files = sorted(fake_audio_dir.glob("*.wav"))

n_created = 0
for audio_file in real_files:
    save_path = real_spec_dir / (audio_file.stem + ".png")
    ok = create_spectrogram(audio_file, save_path)
    if ok:
        n_created += 1

for audio_file in fake_files:
    save_path = fake_spec_dir / (audio_file.stem + ".png")
    ok = create_spectrogram(audio_file, save_path)
    if ok:
        n_created += 1

print(f"🎵 Spectrograms created for {n_created} audio files.")
print("✅ All done!")
