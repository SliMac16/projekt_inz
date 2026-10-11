import argparse
import json

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from config import SEED, RESULTS_DIR
from dataset import create_dataloaders
from models import create_model


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@torch.no_grad()
def get_probabilities(model, loader):
    model.eval()

    targets = []
    probabilities = []

    for images, labels in loader:
        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1,
        )[:, 1]

        targets.extend(labels.tolist())
        probabilities.extend(
            probs.cpu().tolist()
        )

    return targets, probabilities


def evaluate_roc(model_name):
    _, _, test_loader = create_dataloaders(SEED)

    model = create_model(model_name).to(DEVICE)

    checkpoint_path = (
        RESULTS_DIR
        / model_name
        / "best_model.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    )

    targets, probabilities = get_probabilities(
        model,
        test_loader,
    )

    auc = roc_auc_score(
        targets,
        probabilities,
    )

    fpr, tpr, _ = roc_curve(
        targets,
        probabilities,
    )

    output_dir = RESULTS_DIR / model_name

    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {model_name}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
    )

    plt.close()

    metrics_path = output_dir / "metrics.json"

    with open(
        metrics_path,
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    metrics["test"]["roc_auc"] = auc

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("=" * 60)
    print(f"Model:    {model_name}")
    print(f"ROC-AUC:  {auc:.4f}")
    print(
        f"Wykres:   "
        f"{output_dir / 'roc_curve.png'}"
    )
    print("=" * 60)


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

    evaluate_roc(args.model)