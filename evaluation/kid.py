from torchmetrics.image.kid import KernelInceptionDistance
import torch
import csv
import os
from PIL import Image
import numpy as np


def load_images_as_tensor(folder, image_size=299):
    images = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB").resize((image_size, image_size))
                img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                images.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    if images:
        return torch.stack(images)
    else:
        return torch.empty((0, 3, image_size, image_size))


real_images = "/home/data/yunsu/SCoFT/culture_data/korea/train/images"
fake_images = ["/home/evaluation/face",
                "/home/evaluation/GQA",
                "/home/evaluation/merged",
                "/home/evaluation/vanila"]

kid_results = []
real_imgs = load_images_as_tensor(real_images)
for fake_dir in fake_images:
    fake_imgs = load_images_as_tensor(fake_dir)
    kid = KernelInceptionDistance(normalize=True, subset_size=1000)
    kid.update(real_imgs, real=True)
    kid.update(fake_imgs, real=False)
    kid_score_mean, kid_score_std = kid.compute()
    print(f"KID for {fake_dir}: {kid_score_mean} (std: {kid_score_std})")
    kid_results.append({"model": os.path.basename(fake_dir), "kid_score": float(kid_score_mean)})
# Save results to CSV
csv_path = "kid_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["model", "kid_score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(kid_results)
print(f"Saved KID results to {csv_path}")
