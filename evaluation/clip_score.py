import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from PIL import Image
import numpy as np
import os
from glob import glob
import csv
from collections import defaultdict

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

test_prompts = [
    "Generate two people wearing traditional clothing, in Korea",
    "Generate Korean male face",
    "Generate Korean female face",
    "Generate Korean young boy face",
    "Generate Korean young girl face",
    "Generate a man wearing traditional clothing next to a traditional house, in Korea",
    "Generate a Korean family portrait, in a traditional house",
    "Generate a Korean traditional food",
    "Generate Kimchi on the right side of Samgyetang",
    "Generate a Korean King",
    "Korean traditional hat",
    "Kimchi stew"
]

test_prompts_korea = [
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
    "Korea",
]

image_dir = "/home/SCoFT/test_100step"
results = []

def calculate_clip_score(image_path, prompt):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = np.array(image).astype("float32") / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    score = clip_score_fn(image_tensor, [prompt]).detach()
    return round(float(score), 4)

for i, prompt in enumerate(test_prompts_korea, 1):
    pattern = os.path.join(image_dir, f"*_{'prompt'+str(i)}.png")
    for image_path in glob(pattern):
        score = calculate_clip_score(image_path, prompt)
        print(f"{os.path.basename(image_path)} | Prompt: {prompt} | CLIP score: {score}")
        results.append({"image": os.path.basename(image_path), "prompt": prompt, "clip_score": score})

# 결과를 CSV로 저장
csv_path = "clip_score_results_korea.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["image", "prompt", "clip_score"])
    writer.writeheader()
    writer.writerows(results)
print(f"Saved results to {csv_path}")

input_csv = csv_path
output_csv = "{}_pivot".format(os.path.splitext(input_csv)[0])

# 모델별로 prompt별 점수 저장
scores = defaultdict(dict)

with open(input_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 모델명 추출 (예: scoft-Korean-culture_merged_neg)
        image = row["image"]
        if "_prompt" in image:
            model_name = image.rsplit("_prompt", 1)[0]
            prompt_num = image.split("_prompt")[-1].split(".")[0]  # 1~12
            scores[model_name][f"prompt{prompt_num}"] = row["clip_score"]

# 열 순서
prompt_cols = [f"prompt{i}" for i in range(1, 13)]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["model"] + prompt_cols)
    for model in sorted(scores.keys()):
        row = [model] + [scores[model].get(col, "") for col in prompt_cols]
        writer.writerow(row)

print(f"Saved pivoted csv to {output_csv}")