from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicMarginLoss(nn.Module):
    def __init__(self, class_counts, alpha: float = 0.35, scale: float = 20.0):
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        margins = alpha / torch.log1p(counts + 1.0)
        self.register_buffer("margins", margins)
        self.scale = scale

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_margins = self.margins[targets].to(logits.device)
        adjusted = logits.clone()
        adjusted[torch.arange(logits.size(0), device=logits.device), targets] -= target_margins
        return F.cross_entropy(self.scale * adjusted, targets)


class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        costs = self.cost_matrix[targets].to(logits.device)
        return torch.sum(costs * probs, dim=1).mean()


class RecallAwareLoss(nn.Module):
    def __init__(self, high_risk_classes, high_risk_weight: float = 2.0):
        super().__init__()
        self.high_risk_classes = set(int(x) for x in high_risk_classes)
        self.high_risk_weight = high_risk_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        weights = torch.ones_like(ce)
        for c in self.high_risk_classes:
            weights = torch.where(targets == c, torch.full_like(weights, self.high_risk_weight), weights)
        return (ce * weights).mean()


class BiodramCompositeLoss(nn.Module):
    def __init__(self, class_counts, cost_matrix, config):
        super().__init__()
        loss_cfg = config["loss"]
        self.lambda_ce = float(loss_cfg.get("lambda_ce", 1.0))
        self.lambda_margin = float(loss_cfg.get("lambda_margin", 0.2))
        self.lambda_cost = float(loss_cfg.get("lambda_cost", 0.2))
        self.lambda_recall = float(loss_cfg.get("lambda_recall", 0.2))
        self.ce = nn.CrossEntropyLoss()
        self.margin = DynamicMarginLoss(class_counts, loss_cfg.get("margin_alpha", 0.35), loss_cfg.get("logit_scale", 20.0))
        self.cost = CostSensitiveLoss(cost_matrix)
        self.recall = RecallAwareLoss(loss_cfg.get("high_risk_classes", []), loss_cfg.get("high_risk_weight", 2.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.lambda_ce * self.ce(logits, targets)
            + self.lambda_margin * self.margin(logits, targets)
            + self.lambda_cost * self.cost(logits, targets)
            + self.lambda_recall * self.recall(logits, targets)
        )
