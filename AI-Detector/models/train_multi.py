"""
train_multi.py

Train Image (MobileNetV2) and Audio (CNN on spectrograms) models separately.
Video inference uses the trained Image model by extracting frames.

Dataset structure:

Images:
datasets/images/
    train/
        real/
        fake/
    val/
        real/
        fake/

Audio:
datasets/audio/
    train/
        real/
        fake/
    val/
        real/
        fake/
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report

from PIL import Image, UnidentifiedImageError
import torchaudio
import numpy as np
import cv2

# ---------------------------
# Utils
# ---------------------------
def safe_image_loader(path):
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError):
        # Return blank image if corrupted
        return Image.new("RGB", (224,224))

# Audio dataset loader
class AudioDataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = classes or sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls:i for i,cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = self.root_dir/cls
            for file in cls_path.glob("*.*"):
                if file.suffix.lower() in [".wav",".mp3",".flac",".m4a"]:
                    self.samples.append((str(file), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path, backend="soundfile")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64)(waveform)
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        # Resize to 224x224
        mel_spec = F.interpolate(mel_spec.unsqueeze(0), size=(224,224), mode='bilinear', align_corners=False)[0]
        # 3 channels
        mel_spec = mel_spec.repeat(3,1,1)
        # Normalize
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        mel_spec = (mel_spec - mean) / (std + 1e-6)

        return mel_spec, label

# ---------------------------
# Models
# ---------------------------
def build_image_model(num_classes=2, device='cpu'):
    model = models.mobilenet_v2(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    model.to(device)
    return model

def build_audio_model(num_classes=2, device='cpu'):
    # Simple CNN for audio spectrograms
    class AudioCNN(nn.Module):
        def __init__(self, num_classes=2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((7,7))
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64*7*7,128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128,num_classes)
            )
        def forward(self,x):
            return self.fc(self.conv(x))
    model = AudioCNN(num_classes=num_classes).to(device)
    return model

# ---------------------------
# Training Functions
# ---------------------------
def train_model(model, dataloaders, device, epochs=5, lr=1e-4, save_path="model.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_loss = sum(train_losses)/len(train_losses)

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({'model_state_dict':model.state_dict(), 'classes':dataloaders['classes']}, save_path)
            print(f"✅ Saved best model to {save_path}")
    return model

# ---------------------------
# Video Classification
# ---------------------------
def classify_video(video_path, model, device='cpu', preprocess=None):
    if preprocess is None:
        preprocess = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    cap = cv2.VideoCapture(video_path)
    frame_preds = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            frame_preds.append(pred)
    cap.release()
    counts = np.bincount(frame_preds)
    pred_idx = int(np.argmax(counts))
    return pred_idx, frame_preds

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="datasets/images")
    parser.add_argument("--audio-dir", type=str, default="datasets/audio")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--save-image-model", type=str, default="image_model.pth")
    parser.add_argument("--save-audio-model", type=str, default="audio_model.pth")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ----------- Image Training -----------
    print("\n📷 Training Image Model...")
    train_ds = datasets.ImageFolder(os.path.join(args.image_dir,"train"), transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]), loader=safe_image_loader)
    val_ds = datasets.ImageFolder(os.path.join(args.image_dir,"val"), transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]), loader=safe_image_loader)
    image_dataloaders = {
        'train': DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=args.batch_size, shuffle=False),
        'classes': train_ds.classes
    }
    image_model = build_image_model(num_classes=len(train_ds.classes), device=device)
    image_model = train_model(image_model, image_dataloaders, device, epochs=args.epochs, lr=args.lr, save_path=args.save_image_model)

    # ----------- Audio Training -----------
    print("\n🎵 Training Audio Model...")
    audio_train = AudioDataset(os.path.join(args.audio_dir,"train"))
    audio_val = AudioDataset(os.path.join(args.audio_dir,"val"))
    audio_dataloaders = {
        'train': DataLoader(audio_train, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(audio_val, batch_size=args.batch_size, shuffle=False),
        'classes': audio_train.classes
    }
    audio_model = build_audio_model(num_classes=len(audio_train.classes), device=device)
    audio_model = train_model(audio_model, audio_dataloaders, device, epochs=args.epochs, lr=args.lr, save_path=args.save_audio_model)

    print("\n🎉 All training finished. Models saved!")
