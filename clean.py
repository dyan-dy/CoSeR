import safetensors
from safetensors.torch import load_file, save_file
import re

# 定义一个函数来清理键中的不可见字符
def clean_key(key):
    return re.sub(r'[\xa0\xd8]', '', key)

# 加载 safetensors 文件
checkpoint_path = "/home/gdj592/code/3dv/CoSeR/input/vae/diffusion_pytorch_model.safetensors"
tensors = load_file(checkpoint_path)

# 创建一个新的字典来保存清理后的键和值
cleaned_tensors = {}
for key, value in tensors.items():
    new_key = clean_key(key)
    cleaned_tensors[new_key] = value

# 保存清理后的 tensor 到一个新的 safetensors 文件中
cleaned_checkpoint_path = "/home/gdj592/code/3dv/CoSeR/input/vae/diffusion_pytorch_model_clean.safetensors"
save_file(cleaned_tensors, cleaned_checkpoint_path)
