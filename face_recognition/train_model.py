import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .model import CosModel


class CosFaceLoss(nn.Module):
    def __init__(self, m, s, n_classes):
        super(CosFaceLoss, self).__init__()
        self.m = m
        self.s = s
        self.n_classes = n_classes

    def forward(self, cos_theta, y):
        y_oh = F.one_hot(y, self.n_classes)
        cos_theta_m = cos_theta - self.m * y_oh
        logits = cos_theta_m * self.s
        return F.cross_entropy(logits, y)


class MyTrainingModule(pl.LightningModule):
    def __init__(self, lr, weight_decay, s, m, n_classes, output_dim, step_size, gamma):
        super().__init__()
        self.model = CosModel(n_classes, output_dim)
        self.loss = CosFaceLoss(s, m, n_classes)
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

    def training_step(self, batch, batch_idx):
        """The full training loop"""
        X_val, Y_val = batch["image"], batch["label"]
        features, cos_theta = self.model(X_val)
        loss = self.loss(cos_theta, Y_val)

        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        """Define optimizers and LR schedulers."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        lr_dict = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }

        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):
        X_val, Y_val = batch["image"], batch["label"]
        features, cos_theta = self.model(X_val)
        loss = self.loss(cos_theta, Y_val)

        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=False, on_epoch=True, logger=True)

        return metrics


def train_model(cfg, train_loader, val_loader, device):
    trainer = pl.Trainer(
        max_epochs=cfg["epochs_number"],
        accelerator=device,
        enable_checkpointing=False,
        logger=False,
    )
    training_module = MyTrainingModule(
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        s=cfg["cos_face_params"]["s"],
        m=cfg["cos_face_params"]["m"],
        n_classes=cfg["n_classes"],
        output_dim=cfg["output_dim"],
        step_size=cfg["scheduler_params"]["step_size"],
        gamma=cfg["scheduler_params"]["step_size"],
    )

    trainer.fit(training_module, train_loader, val_loader)
    return training_module.model
