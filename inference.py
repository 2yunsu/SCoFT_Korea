import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
from tqdm import tqdm

model_base = "CompVis/stable-diffusion-v1-4"
lora_model_paths = [
    "/home/data/yunsu/SCoFT/output/scoft-Korean-culture_face_neg",
    "/home/data/yunsu/SCoFT/output/scoft-Korean-culture_GQA_neg",
    "/home/data/yunsu/SCoFT/output/scoft-Korean-culture_merged_neg",
    "/home/data/yunsu/SCoFT/output/scoft-Korean-culture_vanila_neg",
]

output_dir = "/home/SCoFT/test_100step"
os.makedirs(output_dir, exist_ok=True)

test_prompts = [
    "Generate two people wearing traditional clothing, in Korea",
    "Generate Korean male face",
    "Generate Korean female face",
    "Generate Korean young boy face",
    "Generate Korean young girl face",
    "Generate a man wearing traditional clothing next to a traditional house, in Korea",
    "Generate a Korean family portrait, in a traditional house",
    "Generate a Korean traditional food",
    "Generate Kimchi on the right side of Samgyetang",
    "Generate a Korean King"
    "Korean traditional hat",
    "Kimchi stew"
]

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

for lora_path in tqdm(lora_model_paths):
    pipe.unet.load_attn_procs(lora_path)
    for idx, prompt in enumerate(test_prompts):
        generator = torch.Generator(device="cuda").manual_seed(10000 + idx)
        image = pipe(
            prompt,
            generator=generator,
            num_inference_steps=50,
            guidance_scale=4.5,
            cross_attention_kwargs={"scale": 1.0}
        ).images[0]
        lora_name = os.path.basename(lora_path)
        image.save(os.path.join(output_dir, f"{lora_name}_prompt{idx+1}.png"))