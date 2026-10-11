from pathlib import Path
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
from PIL import Image


# ============================================================
# KONFIGURACJA
# ============================================================

MANIFEST = Path("final_dataset_manifest.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TRANSFORMACJE
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


# ============================================================
# DATASET
# ============================================================

class MicrostructureDataset(Dataset):

    def __init__(
        self,
        rows,
        transform=None
    ):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(
        self,
        index
    ):

        row = self.rows[index]

        # Manifest został wygenerowany na Windows,
        # dlatego normalizujemy "\" -> "/"
        image_path = Path(
            row["output_path"]
            .replace("\\", "/")
        )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

        if self.transform:
            image = self.transform(
                image
            )

        label = int(
            row["label"]
        )

        return image, label


# ============================================================
# MANIFEST
# ============================================================

with open(
    MANIFEST,
    "r",
    encoding="utf-8-sig"
) as f:

    rows = list(
        csv.DictReader(f)
    )


train_rows = [
    row for row in rows
    if row["split"] == "train"
]

val_rows = [
    row for row in rows
    if row["split"] == "validation"
]

test_rows = [
    row for row in rows
    if row["split"] == "test"
]


# ============================================================
# DATASETY
# ============================================================

train_dataset = MicrostructureDataset(
    train_rows,
    transform=train_transform
)

val_dataset = MicrostructureDataset(
    val_rows,
    transform=eval_transform
)

test_dataset = MicrostructureDataset(
    test_rows,
    transform=eval_transform
)


# ============================================================
# DATALOADERY
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# STATYSTYKI
# ============================================================

print("=" * 70)
print("KONTROLA DATALOADERÓW")
print("=" * 70)

print(f"Urządzenie:    {DEVICE}")
print(f"Batch size:    {BATCH_SIZE}")
print(f"Rozmiar obrazu: {IMAGE_SIZE}x{IMAGE_SIZE}")

print()

print(
    f"train:       {len(train_dataset)}"
)

print(
    f"validation:  {len(val_dataset)}"
)

print(
    f"test:        {len(test_dataset)}"
)


print("\nROZKŁAD KLAS:")

for name, subset in [
    ("train", train_rows),
    ("validation", val_rows),
    ("test", test_rows),
]:

    clean = sum(
        int(row["label"]) == 0
        for row in subset
    )

    anomaly = sum(
        int(row["label"]) == 1
        for row in subset
    )

    print(
        f"{name:12s}: "
        f"{clean:3d} clean | "
        f"{anomaly:3d} anomaly"
    )


# ============================================================
# TEST BATCHA
# ============================================================

images, labels = next(
    iter(train_loader)
)

print("\nTEST BATCHA:")

print(
    f"images.shape = "
    f"{images.shape}"
)

print(
    f"labels.shape = "
    f"{labels.shape}"
)

print(
    f"labels = "
    f"{labels.tolist()}"
)


# ============================================================
# RESNET-50
# ============================================================

print("\n" + "=" * 70)
print("TEST RESNET-50")
print("=" * 70)

resnet = resnet50(
    weights=ResNet50_Weights.DEFAULT
)

resnet.fc = torch.nn.Linear(
    resnet.fc.in_features,
    2
)

resnet = resnet.to(
    DEVICE
)

resnet.eval()

with torch.no_grad():

    output = resnet(
        images.to(DEVICE)
    )

print(
    f"Output shape: "
    f"{output.shape}"
)

del resnet

torch.cuda.empty_cache()


# ============================================================
# ViT-B/16
# ============================================================

print("\n" + "=" * 70)
print("TEST ViT-B/16")
print("=" * 70)

vit = vit_b_16(
    weights=ViT_B_16_Weights.DEFAULT
)

vit.heads.head = torch.nn.Linear(
    vit.heads.head.in_features,
    2
)

vit = vit.to(
    DEVICE
)

vit.eval()

with torch.no_grad():

    output = vit(
        images.to(DEVICE)
    )

print(
    f"Output shape: "
    f"{output.shape}"
)

del vit

torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("KONTROLA ZAKOŃCZONA POPRAWNIE")
print("=" * 70)