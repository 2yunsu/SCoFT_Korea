import json

input_path = "/home/data/yunsu/face_aging_data/train/metadata.jsonl"
output_path = "/home/data/yunsu/face_aging_data/train/metadata_with_negative.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        if line.strip():
            data = json.loads(line)
            data["negative_imgpath"] = []
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")