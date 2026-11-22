"""Modelos: CNN base desde cero y dos arquitecturas ligeras con transferencia."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """4 bloques conv (32-64-128-256) + global average pooling + dropout + FC."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), conv_block(32, 64),
            conv_block(64, 128), conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name == "mobilenetv2":
        w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        m = models.mobilenet_v2(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Modelo desconocido: {name}")


def head_parameters(model: nn.Module, name: str):
    """Parámetros de la cabeza (para la etapa 1 del ajuste fino por etapas)."""
    if name == "baseline":
        return list(model.parameters())
    return list(model.classifier.parameters())


def gradcam_layer(model: nn.Module, name: str) -> nn.Module:
    """Última capa convolucional, usada por Grad-CAM."""
    if name == "baseline":
        return model.features[-1]
    return model.features[-1]
