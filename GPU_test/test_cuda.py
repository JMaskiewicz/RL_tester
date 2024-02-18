import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")
    return device

print(torch.cuda.is_available())

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check the CUDA version used by the current PyTorch installation
print(f"PyTorch Built with CUDA: {torch.version.cuda}")

# Use the function to get the device
device = get_device()

print(f"PyTorch Version: {torch.__version__}")
print(f"PyTorch Built with CUDA: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")