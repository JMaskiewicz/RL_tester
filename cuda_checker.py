import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("CUDA Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")