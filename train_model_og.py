from pathlib import Path
import argparse
import csv
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)
from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# KONFIGURACJA
# ============================================================

SEED = 42

MANIFEST = Path("final_dataset_manifest.csv")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

HEAD_LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4

OUTPUT_ROOT = Path("training_results")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


# ============================================================
# ARGUMENTY
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=[
        "resnet50",
        "vit_b_16"
    ]
)

args = parser.parse_args()

MODEL_NAME = args.model


# ============================================================
# POWTARZALNOŚĆ
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ============================================================
# TRANSFORMACJE
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


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
        transform
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

        image_path = Path(
            row["output_path"]
            .replace("\\", "/")
        )

        with Image.open(
            image_path
        ) as img:

            image = img.convert(
                "RGB"
            )

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


train_dataset = MicrostructureDataset(
    train_rows,
    train_transform
)

val_dataset = MicrostructureDataset(
    val_rows,
    eval_transform
)

test_dataset = MicrostructureDataset(
    test_rows,
    eval_transform
)


loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    generator=loader_generator
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# MODEL
# ============================================================

def create_model(name):

    if name == "resnet50":

        model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

    elif name == "vit_b_16":

        model = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            2
        )

    else:

        raise ValueError(
            f"Nieznany model: {name}"
        )

    return model


def freeze_backbone(
    model,
    name
):

    for parameter in model.parameters():
        parameter.requires_grad = False

    if name == "resnet50":

        for parameter in (
            model.fc.parameters()
        ):
            parameter.requires_grad = True

    else:

        for parameter in (
            model.heads.parameters()
        ):
            parameter.requires_grad = True


def unfreeze_all(model):

    for parameter in model.parameters():
        parameter.requires_grad = True


model = create_model(
    MODEL_NAME
).to(DEVICE)


# ============================================================
# METRYKI
# ============================================================

def calculate_metrics(
    targets,
    predictions
):

    return {
        "accuracy": accuracy_score(
            targets,
            predictions
        ),

        "precision": precision_score(
            targets,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            targets,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            targets,
            predictions,
            zero_division=0
        ),
    }


# ============================================================
# JEDNA EPOKA TRENINGOWA
# ============================================================

def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []


    progress = tqdm(
        loader,
        desc="train",
        leave=False
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        all_targets.extend(
            labels
            .detach()
            .cpu()
            .tolist()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .tolist()
        )


    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = epoch_loss

    return metrics


# ============================================================
# WALIDACJA / TEST
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []


    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )


        predictions = (
            outputs.argmax(
                dim=1
            )
        )


        running_loss += (
            loss.item()
            * images.size(0)
        )


        all_targets.extend(
            labels.cpu().tolist()
        )

        all_predictions.extend(
            predictions.cpu().tolist()
        )


    metrics = calculate_metrics(
        all_targets,
        all_predictions
    )

    metrics["loss"] = (
        running_loss
        / len(loader.dataset)
    )


    return (
        metrics,
        all_targets,
        all_predictions
    )


# ============================================================
# KATALOG WYNIKÓW
# ============================================================

OUTPUT_DIR = (
    OUTPUT_ROOT
    / MODEL_NAME
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best_model.pt"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


history = []

best_val_f1 = -1.0
best_epoch = 0

global_epoch = 0


# ============================================================
# ETAP 1: GŁOWICA
# ============================================================

print("=" * 80)
print(f"MODEL: {MODEL_NAME}")
print("=" * 80)

print(f"Device:       {DEVICE}")
print(f"AMP:          {USE_AMP}")
print(f"Train:        {len(train_dataset)}")
print(f"Validation:   {len(val_dataset)}")
print(f"Test:         {len(test_dataset)}")


freeze_backbone(
    model,
    MODEL_NAME
)


optimizer = torch.optim.AdamW(
    filter(
        lambda p:
            p.requires_grad,
        model.parameters()
    ),
    lr=HEAD_LR,
    weight_decay=WEIGHT_DECAY
)


scaler = torch.amp.GradScaler(
    DEVICE.type,
    enabled=USE_AMP
)


print("\n" + "=" * 80)
print("ETAP 1 - TRENING GŁOWICY")
print("=" * 80)


for _ in range(HEAD_EPOCHS):

    global_epoch += 1

    start = time.time()


    train_metrics = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler
    )


    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )


    elapsed = (
        time.time()
        - start
    )


    history.append({
        "epoch": global_epoch,
        "phase": "head",
        "train_loss":
            train_metrics["loss"],
        "train_accuracy":
            train_metrics["accuracy"],
        "train_precision":
            train_metrics["precision"],
        "train_recall":
            train_metrics["recall"],
        "train_f1":
            train_metrics["f1"],
        "val_loss":
            val_metrics["loss"],
        "val_accuracy":
            val_metrics["accuracy"],
        "val_precision":
            val_metrics["precision"],
        "val_recall":
            val_metrics["recall"],
        "val_f1":
            val_metrics["f1"],
        "seconds":
            elapsed
    })


    print(
        f"Epoch {global_epoch:02d} | "
        f"train loss {train_metrics['loss']:.4f} | "
        f"train F1 {train_metrics['f1']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"val F1 {val_metrics['f1']:.4f}"
    )


    if (
        val_metrics["f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["f1"]
        )

        best_epoch = global_epoch

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


# ============================================================
# ETAP 2: FINE-TUNING
# ============================================================

unfreeze_all(model)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=FINETUNE_LR,
    weight_decay=WEIGHT_DECAY
)


scaler = torch.amp.GradScaler(
    DEVICE.type,
    enabled=USE_AMP
)


epochs_without_improvement = 0


print("\n" + "=" * 80)
print("ETAP 2 - PEŁNY FINE-TUNING")
print("=" * 80)


for _ in range(FINETUNE_EPOCHS):

    global_epoch += 1

    start = time.time()


    train_metrics = train_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        scaler
    )


    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion
    )


    elapsed = (
        time.time()
        - start
    )


    history.append({
        "epoch": global_epoch,
        "phase": "finetune",
        "train_loss":
            train_metrics["loss"],
        "train_accuracy":
            train_metrics["accuracy"],
        "train_precision":
            train_metrics["precision"],
        "train_recall":
            train_metrics["recall"],
        "train_f1":
            train_metrics["f1"],
        "val_loss":
            val_metrics["loss"],
        "val_accuracy":
            val_metrics["accuracy"],
        "val_precision":
            val_metrics["precision"],
        "val_recall":
            val_metrics["recall"],
        "val_f1":
            val_metrics["f1"],
        "seconds":
            elapsed
    })


    print(
        f"Epoch {global_epoch:02d} | "
        f"train loss {train_metrics['loss']:.4f} | "
        f"train F1 {train_metrics['f1']:.4f} | "
        f"val loss {val_metrics['loss']:.4f} | "
        f"val F1 {val_metrics['f1']:.4f}"
    )


    if (
        val_metrics["f1"]
        > best_val_f1
    ):

        best_val_f1 = (
            val_metrics["f1"]
        )

        best_epoch = global_epoch

        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

    else:

        epochs_without_improvement += 1


    if (
        epochs_without_improvement
        >= EARLY_STOPPING_PATIENCE
    ):

        print(
            "\nEarly stopping."
        )

        break


# ============================================================
# HISTORIA
# ============================================================

history_path = (
    OUTPUT_DIR
    / "history.csv"
)


with open(
    history_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# WCZYTANIE NAJLEPSZEGO MODELU
# ============================================================

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)


# ============================================================
# FINALNA WALIDACJA
# ============================================================

val_metrics, _, _ = evaluate(
    model,
    val_loader,
    criterion
)


# ============================================================
# TEST - DOPIERO TERAZ
# ============================================================

test_metrics, (
    test_targets
), (
    test_predictions
) = evaluate(
    model,
    test_loader,
    criterion
)


cm = confusion_matrix(
    test_targets,
    test_predictions
)


# ============================================================
# ZAPIS METRYK
# ============================================================

results = {
    "model": MODEL_NAME,

    "seed": SEED,

    "best_epoch": best_epoch,

    "best_validation_f1":
        best_val_f1,

    "validation":
        val_metrics,

    "test":
        test_metrics,

    "confusion_matrix":
        cm.tolist(),

    "train_samples":
        len(train_dataset),

    "validation_samples":
        len(val_dataset),

    "test_samples":
        len(test_dataset),

    "batch_size":
        BATCH_SIZE,

    "image_size":
        IMAGE_SIZE,

    "head_epochs":
        HEAD_EPOCHS,

    "max_finetune_epochs":
        FINETUNE_EPOCHS,

    "head_lr":
        HEAD_LR,

    "finetune_lr":
        FINETUNE_LR,

    "weight_decay":
        WEIGHT_DECAY
}


with open(
    OUTPUT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# MACIERZ POMYŁEK
# ============================================================

fig = plt.figure(
    figsize=(5, 5)
)

ax = fig.add_subplot(111)

image = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "clean",
    "anomaly"
])

ax.set_yticklabels([
    "clean",
    "anomaly"
])

ax.set_xlabel(
    "Predykcja"
)

ax.set_ylabel(
    "Klasa rzeczywista"
)

ax.set_title(
    f"Confusion matrix - {MODEL_NAME}"
)


for i in range(2):

    for j in range(2):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )


fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "confusion_matrix.png",
    dpi=200
)

plt.close(fig)


# ============================================================
# KRZYWE TRENINGU
# ============================================================

epochs = [
    row["epoch"]
    for row in history
]

train_loss = [
    row["train_loss"]
    for row in history
]

val_loss = [
    row["val_loss"]
    for row in history
]


fig = plt.figure(
    figsize=(7, 5)
)

ax = fig.add_subplot(111)

ax.plot(
    epochs,
    train_loss,
    label="train"
)

ax.plot(
    epochs,
    val_loss,
    label="validation"
)

ax.set_xlabel(
    "Epoka"
)

ax.set_ylabel(
    "Loss"
)

ax.set_title(
    f"Loss - {MODEL_NAME}"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "loss_curve.png",
    dpi=200
)

plt.close(fig)


train_f1 = [
    row["train_f1"]
    for row in history
]

val_f1 = [
    row["val_f1"]
    for row in history
]


fig = plt.figure(
    figsize=(7, 5)
)

ax = fig.add_subplot(111)

ax.plot(
    epochs,
    train_f1,
    label="train"
)

ax.plot(
    epochs,
    val_f1,
    label="validation"
)

ax.set_xlabel(
    "Epoka"
)

ax.set_ylabel(
    "F1"
)

ax.set_title(
    f"F1 - {MODEL_NAME}"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "f1_curve.png",
    dpi=200
)

plt.close(fig)


# ============================================================
# PODSUMOWANIE
# ============================================================

print("\n" + "=" * 80)
print("WYNIK KOŃCOWY")
print("=" * 80)

print(
    f"Model:               "
    f"{MODEL_NAME}"
)

print(
    f"Najlepsza epoka:     "
    f"{best_epoch}"
)

print(
    f"Validation F1:       "
    f"{val_metrics['f1']:.4f}"
)

print()

print(
    f"Test accuracy:       "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Test precision:      "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Test recall:         "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"Test F1:             "
    f"{test_metrics['f1']:.4f}"
)

print()

print(
    "Confusion matrix:"
)

print(cm)

print(
    f"\nWyniki zapisano w: "
    f"{OUTPUT_DIR}"
)