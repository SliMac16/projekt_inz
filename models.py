import torch.nn as nn

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
)


def create_model(name):
    if name == "resnet50":
        model = resnet50(
            weights=ResNet50_Weights.DEFAULT
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            2,
        )

        return model

    if name == "vit_b_16":
        model = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT
        )

        model.heads.head = nn.Linear(
            model.heads.head.in_features,
            2,
        )

        return model

    raise ValueError(f"Nieznany model: {name}")


def freeze_backbone(model, name):
    for parameter in model.parameters():
        parameter.requires_grad = False

    if name == "resnet50":
        parameters = model.fc.parameters()
    else:
        parameters = model.heads.parameters()

    for parameter in parameters:
        parameter.requires_grad = True


def unfreeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = True