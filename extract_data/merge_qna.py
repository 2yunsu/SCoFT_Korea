import json

input_path = "qa_simple_blip2.jsonl"
output_path = "./metadata.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        new_data = {
            "file_name": data.get("file_name", ""),
        }
        # QnA1 ~ QnA10 생성
        for i in range(1, 11):
            q = data.get(f"question_en_{i}", "")
            a = data.get(f"answer_en_{i}", "")
            new_data[f"QnA{i}"] = f"Q: '{q}', A: '{a}'"
        # negative_imgpath 빈 열 추가
        new_data["negative_imgpath"] = ""
        # 필요하다면 blip2_text 등 다른 열도 추가
        if "blip2_text" in data:
            new_data["blip2_text"] = data["blip2_text"]
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")