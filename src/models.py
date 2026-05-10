from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class TaskDimensions:
    num_categories: int
    num_subcategories: int
    num_cooking_styles: int


class ScratchCNN(nn.Module):
    def __init__(self, task_dims: TaskDimensions, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.category_head = nn.Linear(256, task_dims.num_categories)
        self.subcategory_head = nn.Linear(256, task_dims.num_subcategories)
        self.cooking_style_head = nn.Linear(256, task_dims.num_cooking_styles)
        self.calorie_head = nn.Linear(256, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(self.features(x))
        return {
            "category": self.category_head(features),
            "subcategory": self.subcategory_head(features),
            "cooking_style": self.cooking_style_head(features),
            "calories": self.calorie_head(features).squeeze(1),
        }


class PretrainedResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        task_dims: TaskDimensions,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.category_head = nn.Linear(512, task_dims.num_categories)
        self.subcategory_head = nn.Linear(512, task_dims.num_subcategories)
        self.cooking_style_head = nn.Linear(512, task_dims.num_cooking_styles)
        self.calorie_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.shared(self.backbone(x))
        return {
            "category": self.category_head(features),
            "subcategory": self.subcategory_head(features),
            "cooking_style": self.cooking_style_head(features),
            "calories": self.calorie_head(features).squeeze(1),
        }


def make_task_dims(label_maps: dict[str, dict[str, int]]) -> TaskDimensions:
    return TaskDimensions(
        num_categories=len(label_maps["category"]),
        num_subcategories=len(label_maps["subcategory"]),
        num_cooking_styles=len(label_maps["cooking_style"]),
    )


def build_model(
    model_name: str,
    label_maps: dict[str, dict[str, int]],
    pretrained: bool = True,
) -> nn.Module:
    task_dims = make_task_dims(label_maps)
    if model_name == "scratch_cnn":
        return ScratchCNN(task_dims)
    if model_name == "resnet18_subcategory":
        return PretrainedResNetClassifier(task_dims.num_subcategories, pretrained=pretrained)
    if model_name == "multitask_resnet18":
        return MultiTaskModel(task_dims, pretrained=pretrained)
    raise ValueError(f"Unknown model name: {model_name}")
