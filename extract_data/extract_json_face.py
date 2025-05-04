import os
import json

# 이미지와 라벨 폴더 경로
images_dir = "/home/data/yunsu/face_aging_data/images"
labels_dir = "/home/data/yunsu/face_aging_data/labels"
output_jsonl = "/home/data/yunsu/face_aging_data/filtered_labels.jsonl"

# 이미지 파일명(확장자 제외) 리스트 생성
image_files = os.listdir(images_dir)
image_basenames = {os.path.splitext(f)[0] for f in image_files if os.path.isfile(os.path.join(images_dir, f))}

with open(output_jsonl, "w", encoding="utf-8") as out_f:
    for json_file in os.listdir(labels_dir):
        if json_file.endswith(".json"):
            basename = os.path.splitext(json_file)[0]
            if basename in image_basenames:
                json_path = os.path.join(labels_dir, json_file)
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                    # "text" 열 추가: "'age_past' 'gender' face"
                    age_past = data.get("age_past", "")
                    gender = data.get("gender", "")
                    data["text"] = f"{age_past} age {gender} face"
                    out_f.write(json.dumps(data, ensure_ascii=False) + "\n")