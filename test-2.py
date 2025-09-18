import torch

print(torch.__version__, torch.version.cuda)  # ควรเป็น 2.5.1+cu121 / 12.1
print("CUDA?", torch.cuda.is_available())
