from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tlc.client.torch.metrics import MetricsCollector
from tlc.core.schema import Schema
from utils.dice_score import dice_loss


class LossCollector(MetricsCollector):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_metrics(self, batch, predictor_output) -> dict[str, Any]:
        predictions = predictor_output.forward
        _, masks = batch["image"], batch["mask"]
        masks = masks.to(self.device)

        unreduced_criterion = nn.BCEWithLogitsLoss(reduction="none")

        unreduced_dice_loss = dice_loss(
            torch.sigmoid(predictions), masks.float(), multiclass=False, reduce_batch_first=False, reduction="none"
        )

        unreduced_bce_loss = unreduced_criterion(predictions, masks.float()).mean((-1, -2))

        loss = unreduced_dice_loss + unreduced_bce_loss

        metrics_batch = {
            "loss": loss.cpu().numpy().squeeze(),
        }
        return metrics_batch

    @property
    def column_schemas(self) -> dict[str, Schema]:
        return {}
