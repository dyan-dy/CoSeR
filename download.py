from diffusers import StableDiffusionPipeline
import os

# 指定保存模型的文件夹路径
save_dir = "/home/gdj592/code/3dv/CoSeR/input/sd"
os.makedirs(save_dir, exist_ok=True)

# 下载并保存模型
pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", cache_dir=save_dir)
pipeline.save_pretrained(save_dir)
