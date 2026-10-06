"""将 pytorch_model.bin 转换为 model.safetensors（mmap 流式，省内存，绕过 torch>=2.6 限制）"""
import torch
from safetensors.torch import save_file
from pathlib import Path

model_dir = Path(r"C:\Users\cyh\Desktop\111\rag_roleplay\models\bge-m3")
bin_path = model_dir / "pytorch_model.bin"
out_path = model_dir / "model.safetensors"

print("loading with mmap + weights_only ...")
state_dict = torch.load(str(bin_path), map_location="cpu", mmap=True, weights_only=True)

tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
print("tensor count:", len(tensors))

save_file(tensors, str(out_path))
print("saved:", out_path, "size MB:", round(out_path.stat().st_size / 1e6, 1))
