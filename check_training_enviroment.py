import sys
import torch
import torchvision

print("=" * 70)
print("ŚRODOWISKO TRENINGOWE")
print("=" * 70)

print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"Torchvision:  {torchvision.__version__}")

print()
print(f"CUDA dostępna: {torch.cuda.is_available()}")

if torch.cuda.is_available():

    print(
        f"Wersja CUDA:   "
        f"{torch.version.cuda}"
    )

    print(
        f"GPU:           "
        f"{torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM:          "
        f"{total_memory:.2f} GB"
    )

else:

    print(
        "GPU CUDA nie jest dostępne. "
        "Trening będzie wykonywany na CPU."
    )


print("\n" + "=" * 70)
print("TEST MODELI TORCHVISION")
print("=" * 70)

try:

    from torchvision.models import (
        resnet50,
        ResNet50_Weights,
        vit_b_16,
        ViT_B_16_Weights
    )

    print("ResNet-50: OK")
    print("ViT-B/16:  OK")

except Exception as e:

    print("BŁĄD:")
    print(e)


print("\n" + "=" * 70)
print("GOTOWE")
print("=" * 70)