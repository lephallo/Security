# infer_audio_spectrogram.py
import torch
from torchvision import transforms, models
from PIL import Image
import sys

IMG = sys.argv[1]  # path to spectrogram PNG
MODEL = "models/audio_mobilenetv2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

ckpt = torch.load(MODEL, map_location=device)
classes = ckpt['classes']

model = models.mobilenet_v2(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_features, 2))
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

img = Image.open(IMG).convert("RGB")
x = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

print("classes:", classes)
print("probs:", probs.tolist())
print("predicted:", classes[int(probs.argmax())])
