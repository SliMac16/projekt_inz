import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import (
    MANIFEST_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


class MicrostructureDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image_path = Path(
            row["output_path"].replace("\\", "/")
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = int(row["label"])

        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])

    return train_transform, eval_transform


def load_manifest():
    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
    ) as file:
        return list(csv.DictReader(file))


def create_dataloaders(seed):
    rows = load_manifest()

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

    train_transform, eval_transform = get_transforms()

    train_dataset = MicrostructureDataset(
        train_rows,
        train_transform,
    )

    val_dataset = MicrostructureDataset(
        val_rows,
        eval_transform,
    )

    test_dataset = MicrostructureDataset(
        test_rows,
        eval_transform,
    )

    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader