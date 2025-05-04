import os
import json
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# BLIP2 모델 준비
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto")
# model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float16,
#     device_map={"language_model": 0, "vision_model": 0, "qformer": 0}
# )
device = "cuda" if torch.cuda.is_available() else "cpu"

input_jsonl = '/home/data/yunsu/face_aging_data/train/metadata.jsonl'
output_jsonl = '/home/SCoFT/extract_data/face_blip2.jsonl'
img_dir = '/home/data/yunsu/face_aging_data/train/images'

# 처리할 이미지 개수 지정 (예: 100개, None이면 전체)
max_images = None  # 원하는 개수로 변경, 전체 처리시 None

# mixed_data 폴더의 실제 파일명 집합 생성
mixed_files = set(os.listdir(img_dir))

with open(input_jsonl, encoding='utf-8') as fin, open(output_jsonl, 'w', encoding='utf-8') as fout:
    for idx, line in enumerate(fin):
        if max_images is not None and idx >= max_images:
            break
        item = json.loads(line)
        file_name = os.path.basename(item['file_name'])
        if file_name in mixed_files:
            img_path = os.path.join(img_dir, file_name)
            try:
                raw_image = Image.open(img_path).convert('RGB')
                question = ""
                inputs = processor(raw_image, question, return_tensors="pt").to(device)
                out = model.generate(**inputs)
                blip2_text = processor.decode(out[0], skip_special_tokens=True).strip()
            except Exception as e:
                blip2_text = ""
            item['blip2_text'] = blip2_text
        else:
            item['blip2_text'] = ""
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')