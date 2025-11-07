import os
import requests
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Step 1: Create folder structure
# ----------------------------
base_dir = "dataset"
folders = [
    "train/real",
    "train/fake",
    "test/real",
    "test/fake"
]

for folder in folders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# ----------------------------
# Step 2: Download sample images (JPEG only)
# ----------------------------
# Real faces (public domain)
real_urls = [
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg"
]

# Fake faces (AI-generated)
fake_urls = [
    "https://thispersondoesnotexist.com/image",  # sometimes fails
    "https://thispersondoesnotexist.com/image"
]

def download_and_convert_to_jpg(urls, save_folder):
    for i, url in enumerate(urls):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                temp_path = os.path.join(save_folder, f"temp_{i}.jpg")
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                # Convert to RGB JPEG
                try:
                    img = Image.open(temp_path).convert("RGB")
                    final_path = os.path.join(save_folder, f"img{i+1}.jpg")
                    img.save(final_path, "JPEG")
                    os.remove(temp_path)
                    print(f"Downloaded and saved {final_path}")
                except Exception as e:
                    os.remove(temp_path)
                    print(f"Skipping broken image {url}: {e}")
            else:
                print(f"Failed to download {url}, status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

# Download images
download_and_convert_to_jpg(real_urls, os.path.join(base_dir, "train/real"))
download_and_convert_to_jpg(fake_urls, os.path.join(base_dir, "train/fake"))
download_and_convert_to_jpg(real_urls, os.path.join(base_dir, "test/real"))
download_and_convert_to_jpg(fake_urls, os.path.join(base_dir, "test/fake"))

# ----------------------------
# Step 3: Prepare DataLoaders
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(os.path.join(base_dir, "train"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(base_dir, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# ----------------------------
# Step 4: Define a simple CNN
# ----------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*16*16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes: real/fake
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ----------------------------
# Step 5: Train the model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 3

for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ----------------------------
# Step 6: Test the model
# ----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100*correct/total:.2f}%")
