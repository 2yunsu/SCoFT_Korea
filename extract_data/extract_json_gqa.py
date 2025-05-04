import json
import os

# 1) qa.json 경로
qa_path = '/home/data/yunsu/034.한국어_GQA_데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_LABEL_질의응답_AI Hub 업로드/qa.json'
# 2) mixed_data 폴더 경로
mixed_dir = '/home/data/yunsu/034.한국어_GQA_데이터/3.개방데이터/1.데이터/Training/01.raw_data/mixed_data'
# 3) 출력할 QA jsonl 경로
output_qa = 'filtered_qa.jsonl'

# --- QA 로드 & ID 매핑 ---
with open(qa_path, encoding='utf-8') as f:
    qa_list = json.load(f)
qa_map = {item['Scene_Graph_ID']: item for item in qa_list}
scene_ids = set(qa_map.keys())

# --- mixed_data의 .jpg 파일명(stem) 모으기 ---
mixed_ids = {
    os.path.splitext(fn)[0]
    for fn in os.listdir(mixed_dir)
    if fn.lower().endswith('.jpg')
}

# --- 교집합만 추출해서 QA jsonl로 저장 ---
with open(output_qa, 'w', encoding='utf-8') as fout:
    for scene_id in mixed_ids & scene_ids:
        fout.write(json.dumps(qa_map[scene_id], ensure_ascii=False) + '\n')