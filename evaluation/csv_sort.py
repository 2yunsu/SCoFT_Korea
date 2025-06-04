import csv
from collections import defaultdict

input_csv = "clip_score_results_korea.csv"
output_csv = "clip_score_pivot_korea.csv"

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