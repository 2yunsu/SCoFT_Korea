import json
import os

input_path = "/home/data/yunsu/combined_data/train/metadata.jsonl"
output_path = "/home/data/yunsu/combined_data/train/metadata_with_negative.jsonl"
num_negatives = 5

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        item = json.loads(line)
        file_name = item.get("file_name")
        if file_name:
            base_name, ext = os.path.splitext(os.path.basename(file_name))
            ext = ext.lower()
            negative_paths = [
                f'negative_images/{base_name}_negative{i}{ext}' for i in range(1, num_negatives + 1)
            ]
            item["negative_imgpath"] = negative_paths
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")