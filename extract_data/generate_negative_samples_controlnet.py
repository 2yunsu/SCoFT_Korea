import os
import json
from PIL import Image
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm

import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

input_dir = "/home/data/yunsu/combined_data/train/images"
output_dir = "/home/data/yunsu/combined_data/train/negative_images"
# metadata_path = "/home/data/yunsu/combined_data/train/metadata.jsonl"
metadata_path = "/home/data/yunsu/face_aging_data/train/metadata.jsonl"
# metadata_path = "/home/data/yunsu/034.한국어_GQA_데이터/3.개방데이터/1.데이터/Training/culture_data/korea/train/metadata.jsonl"
os.makedirs(output_dir, exist_ok=True)
seed = 1024
num_images = 5 

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda:1")

# metadata.jsonl에서 이미지 파일명과 text 매핑
filename_to_text = {}
with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        filename = item.get("file_name")  # "image" 필드에 파일명
        text = item.get("blip2_text")
        if filename and text:
            filename_to_text[os.path.basename(filename)] = text

for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        prompt = filename_to_text.get(filename)
        if not prompt:
            continue
        init_image = Image.open(os.path.join(input_dir, filename)).convert("RGB")
        name, _ = os.path.splitext(filename)
        for i in range(1, num_images + 1):
            # run inference
            if seed is not None:
                generator = seed + hash(filename) % 100000 + i
            images = []
            prompt_kr = prompt + "in Korea"
            width, height = init_image.size
            if width > 512 or height > 512:
                init_image = init_image.resize((512, 512), Image.LANCZOS)
                width, height = init_image.size
            image = pipe(
                prompt_kr, 
                control_image=init_image,
                width=width,
                height=height,
                controlnet_conditioning_scale=0.7,
                control_guidance_end=0.8,
                num_inference_steps=20, 
                guidance_scale=3.5,
                generator=torch.Generator(device="cuda:1").manual_seed(generator),
            ).images[0]
            save_path = os.path.join(output_dir, f"{name}_negative{i}.jpg")
            image.save(save_path)
