import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: GPU")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Device: CPU")

print("Selected device:", device)

# test
x = torch.randn(500, 1000).to(device)
print("Tensor device:", x.device)