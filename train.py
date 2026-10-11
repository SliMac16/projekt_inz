import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import (
    SEED,
    RESULTS_DIR,
    HEAD_EPOCHS,
    FINETUNE_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    HEAD_LR,
    FINETUNE_LR,
    WEIGHT_DECAY,
)

from dataset import create_dataloaders
from models import (
    create_model,
    freeze_backbone,
    unfreeze_model,
)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_metrics(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(
            targets,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            targets,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            targets,
            predictions,
            zero_division=0,
        ),
    }


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()

    total_loss = 0.0
    targets = []
    predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)

        targets.extend(labels.cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    targets = []
    predictions = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        predicted = outputs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)

        targets.extend(labels.cpu().tolist())
        predictions.extend(predicted.cpu().tolist())

    metrics = calculate_metrics(targets, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics, targets, predictions


def save_history(history, output_dir):
    path = output_dir / "history.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


def run_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scaler,
    epochs,
    phase,
    start_epoch,
    best_f1,
    best_epoch,
    checkpoint_path,
    patience=None,
):
    history = []
    no_improvement = 0
    current_epoch = start_epoch

    for _ in range(epochs):
        current_epoch += 1

        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        history.append({
            "epoch": current_epoch,
            "phase": phase,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })

        print(
            f"Epoch {current_epoch:02d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train F1 {train_metrics['f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val F1 {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = current_epoch
            no_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )
        else:
            no_improvement += 1

        if patience is not None and no_improvement >= patience:
            print("\nEarly stopping.")
            break

    return (
        history,
        current_epoch,
        best_f1,
        best_epoch,
    )


def main(model_name):
    set_seed(SEED)

    train_loader, val_loader, test_loader = (
        create_dataloaders(SEED)
    )

    output_dir = RESULTS_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "best_model.pt"

    model = create_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    print(f"Device:       {DEVICE}")
    print(f"AMP:          {USE_AMP}")
    print(f"Train:        {len(train_loader.dataset)}")
    print(f"Validation:   {len(val_loader.dataset)}")
    print(f"Test:         {len(test_loader.dataset)}")

    history = []
    best_f1 = -1.0
    best_epoch = 0
    current_epoch = 0

    # --------------------------------------------------------
    # Etap 1 - trening głowicy klasyfikacyjnej
    # --------------------------------------------------------

    freeze_backbone(model, model_name)

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=HEAD_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        DEVICE.type,
        enabled=USE_AMP,
    )

    print("\n" + "=" * 70)
    print("ETAP 1 - TRENING GŁOWICY")
    print("=" * 70)

    stage_history, current_epoch, best_f1, best_epoch = (
        run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epochs=HEAD_EPOCHS,
            phase="head",
            start_epoch=current_epoch,
            best_f1=best_f1,
            best_epoch=best_epoch,
            checkpoint_path=checkpoint_path,
        )
    )

    history.extend(stage_history)

    # --------------------------------------------------------
    # Etap 2 - pełny fine-tuning
    # --------------------------------------------------------

    unfreeze_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        DEVICE.type,
        enabled=USE_AMP,
    )

    print("\n" + "=" * 70)
    print("ETAP 2 - PEŁNY FINE-TUNING")
    print("=" * 70)

    stage_history, current_epoch, best_f1, best_epoch = (
        run_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            epochs=FINETUNE_EPOCHS,
            phase="finetune",
            start_epoch=current_epoch,
            best_f1=best_f1,
            best_epoch=best_epoch,
            checkpoint_path=checkpoint_path,
            patience=EARLY_STOPPING_PATIENCE,
        )
    )

    history.extend(stage_history)

    save_history(history, output_dir)

    # --------------------------------------------------------
    # Najlepszy checkpoint
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    )

    val_metrics, _, _ = evaluate(
        model,
        val_loader,
        criterion,
    )

    # Test wykonywany dopiero po zakończeniu treningu
    test_metrics, targets, predictions = evaluate(
        model,
        test_loader,
        criterion,
    )

    cm = confusion_matrix(
        targets,
        predictions,
    )

    results = {
        "model": model_name,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "validation": val_metrics,
        "test": test_metrics,
        "confusion_matrix": cm.tolist(),
    }

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("WYNIK KOŃCOWY")
    print("=" * 70)

    print(f"Model:          {model_name}")
    print(f"Najlepsza epoka: {best_epoch}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")

    print()
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    print(
        f"\nWyniki zapisano w: {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "resnet50",
            "vit_b_16",
        ],
    )

    args = parser.parse_args()

    main(args.model)