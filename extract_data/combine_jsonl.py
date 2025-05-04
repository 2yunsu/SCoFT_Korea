import json

# 세 파일 경로
files = [
    "/home/data/yunsu/face_aging_data/train/metadata.jsonl",
    "/home/data/yunsu/034.한국어_GQA_데이터/3.개방데이터/1.데이터/Training/culture_data/korea/train/metadata.jsonl",
    "/home/data/yunsu/SCoFT/culture_data/korea/train/metadata.jsonl"
]
output_path = "/home/SCoFT/extract_data/combined_metadata.jsonl"

def extract_fields(d):
    # file_name
    file_name = d.get("file_name")
    # text
    text = d.get("text") or d.get("QnA1")
    # blip_text
    blip_text = d.get("blip2_text")
    # negative_imgpath: 반드시 리스트로 변환
    negative_imgpath = d.get("negative_imgpath", [])
    if not isinstance(negative_imgpath, list):
        negative_imgpath = [negative_imgpath] if negative_imgpath else []
    return {
        "file_name": file_name,
        "text": text,
        "blip_text": blip_text,
        "negative_imgpath": negative_imgpath
    }

with open(output_path, "w", encoding="utf-8") as out_f:
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "[]" or line == "{}":
                    continue
                try:
                    d = json.loads(line)
                    merged = extract_fields(d)
                    # file_name, text, blip_text가 모두 None이면 건너뜀
                    if not (merged["file_name"] and merged["text"] and merged["blip_text"]):
                        continue
                    out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
                except Exception:
                    continue