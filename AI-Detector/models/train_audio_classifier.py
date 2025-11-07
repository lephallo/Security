import os
import random
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def main():
    # ---------------- CONFIG ----------------
    DATA_ROOT = Path("datasets/audio_spectrograms")  # real/ fake/
    WORK_DIR = Path("models")
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 16
    IMG_SIZE = 224
    EPOCHS = 12
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANDOM_SEED = 42
    TRAIN_VAL_SPLIT = 0.8
    MODEL_SAVE = WORK_DIR / "audio_mobilenetv2.pth"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # ---------------- Prepare dataset ----------------
    def prepare_splits(root: Path, split=TRAIN_VAL_SPLIT):
        train_dir = root / "train"
        val_dir = root / "val"
        if train_dir.exists() and val_dir.exists():
            print("Using existing train/ and val/ folders.")
            return train_dir, val_dir

        print("Creating train/val split from", root)
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        for cls in ["real", "fake"]:
            src = root / cls
            if not src.exists():
                continue
            imgs = sorted([p for p in src.glob("*.png")])
            random.shuffle(imgs)
            cut = int(len(imgs) * split)
            dst_train = train_dir / cls
            dst_val = val_dir / cls
            dst_train.mkdir(parents=True, exist_ok=True)
            dst_val.mkdir(parents=True, exist_ok=True)
            for p in imgs[:cut]:
                shutil.copy2(p, dst_train / p.name)
            for p in imgs[cut:]:
                shutil.copy2(p, dst_val / p.name)
        return train_dir, val_dir

    train_dir, val_dir = prepare_splits(DATA_ROOT)

    # ---------------- Transforms & DataLoaders ----------------
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Classes: {train_ds.classes}")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # ---------------- Model ----------------
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 2)  # 2 classes: real/fake
    )
    model = model.to(DEVICE)

    # ---------------- Loss & Optimizer ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    # ---------------- Training Loop ----------------
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses, y_true_train, y_pred_train = [], [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred_train.extend(preds.tolist())
            y_true_train.extend(yb.detach().cpu().numpy().tolist())

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses, y_true_val, y_pred_val, val_probs = [], [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                y_pred_val.extend(preds.cpu().numpy().tolist())
                y_true_val.extend(yb.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())

        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"time={elapsed:.1f}s  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'classes': train_ds.classes
            }, MODEL_SAVE)
            print(f"Saved best model to {MODEL_SAVE} (val_acc={val_acc:.4f})")

    # ---------------- Final Evaluation with Probabilities ----------------
    print("Loading best model for final evaluation...")
    ckpt = torch.load(MODEL_SAVE, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    print("Final val accuracy:", accuracy_score(all_labels, all_preds))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("Classification report:")
    print(classification_report(all_labels, all_preds, target_names=ckpt['classes']))

    # Print first 5 probability predictions as example
    for i in range(min(5, len(all_probs))):
        print(f"Sample {i+1}: Label={ckpt['classes'][all_preds[i]]}, Probabilities={all_probs[i]}")

if __name__ == "__main__":
    main()
