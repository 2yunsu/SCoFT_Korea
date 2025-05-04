import os
import json
from PIL import Image

input_dir = "/home/data/yunsu/face_aging_data/train/images_raw"
output_dir = "/home/data/yunsu/face_aging_data/train/imagesp"
metadata_path = "/home/data/yunsu/face_aging_data/train/metadata.jsonl"
os.makedirs(output_dir, exist_ok=True)

# 파일명과 box 정보 매핑
filename_to_box = {}
with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        filename = os.path.basename(item.get("file_name"))
        annotation = item.get("annotation")
        if filename and annotation and len(annotation) > 0:
            box = annotation[0].get("box")
            if box:
                filename_to_box[filename] = box

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        box = filename_to_box.get(filename)
        if not box:
            continue
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path).convert("RGB")
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        # crop 영역을 정수로 변환
        left = int(round(x))
        upper = int(round(y))
        right = int(round(x + w))
        lower = int(round(y + h))
        cropped = image.crop((left, upper, right, lower))
        cropped.save(os.path.join(output_dir, filename))