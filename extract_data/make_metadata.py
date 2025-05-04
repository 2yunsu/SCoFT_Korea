import json

input_path = '/home/SCoFT/culture_data/korea/train/filtered_qa.jsonl'
output_path = 'qa_simple.jsonl'

with open(input_path, encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        qa = json.loads(line)
        scene_id = qa['Scene_Graph_ID']
        file_name = f"mixed_data/{scene_id}.jpg"
        entry = {"file_name": file_name}
        qa_list = qa.get('QA_list', [])
        for idx, qa_item in enumerate(qa_list, 1):
            entry[f"question_en_{idx}"] = qa_item.get('question_en', '')
            entry[f"answer_en_{idx}"] = qa_item.get('answer_en', '')
        fout.write(json.dumps(entry, ensure_ascii=False) + '\n')