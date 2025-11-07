from PIL import Image
import os

base_dir = "dataset"

for split in ["train", "test"]:
    for cls in ["real", "fake"]:
        folder = os.path.join(base_dir, split, cls)
        for filename in os.listdir(folder):
            if filename.lower().endswith(".png"):
                png_path = os.path.join(folder, filename)
                jpg_path = os.path.join(folder, filename.rsplit(".",1)[0] + ".jpg")
                try:
                    im = Image.open(png_path).convert("RGB")
                    im.save(jpg_path, "JPEG")
                    os.remove(png_path)  # remove old PNG
                    print(f"Converted {png_path} -> {jpg_path}")
                except Exception as e:
                    print(f"Error converting {png_path}: {e}")
