#train.py
from argparse import ArgumentParser
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

from evoformer import Evoformer, PositionalEncoding
from roko_data_module import RokoDataModule

POSITIONAL_FEATURES = 5
READ_FEATURES = 12
OUTPUT_CLASSES = 5
MASK_CLASSES = 5


class GRUNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 dropout: float = 0.0) -> None:
        super().__init__()

        self.linear = nn.Linear(POSITIONAL_FEATURES, input_dim)
        self.gru = nn.GRU(input_dim,
                          hidden_dim,
                          n_layers,
                          dropout=dropout,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = F.relu(x)
        x, _ = self.gru(x)

        return x  # BxNxSx2H


class AttentionNetwork(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 nheads: int,
                 nblocks: int,
                 depth_prob: float = 1.0,
                 pos_dropout: float = 0.0) -> None:
        super().__init__()
        # Model

        self.embedding = nn.Embedding(12, embed_dim)
        self.pe = PositionalEncoding(embed_dim, pos_dropout)

        self.evoformer = Evoformer(msa_embedding_dim=embed_dim,
                                   heads=nheads,
                                   num_blocks=nblocks,
                                   p_keep_lowest=depth_prob)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:  # x B R S, pos_stat B S F
        x = self.embedding(x)
        x = self.pe(x)  # B R S E
        x = self.evoformer(x)  # B R S E

        return x


class Polisher(pl.LightningModule):

    def __init__(self,
                 gru_input_dim: int = 128,
                 gru_hidden_dim: int = 128,
                 gru_n_layers: int = 1,
                 gru_dropout: float = 0.0,
                 attn_embed_dim: int = 128,
                 attn_n_heads: int = 8,
                 attn_n_blocks: int = 8,
                 attn_depth_prob: float = 1.0,
                 attn_pos_dropout: float = 0.1,
                 reads_mask_prob: float = 0.2,
                 alpha: float = 0.1,
                 lr=3e-4) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.gru_net = GRUNetwork(gru_input_dim, gru_hidden_dim, gru_n_layers,
                                  gru_dropout)
        self.attn_net = AttentionNetwork(attn_embed_dim, attn_n_heads,
                                         attn_n_blocks, attn_depth_prob,
                                         attn_pos_dropout)

        self.mask_fc = nn.Linear(attn_embed_dim, MASK_CLASSES)
        self.fc = nn.Linear(2 * gru_hidden_dim + attn_embed_dim,
                            OUTPUT_CLASSES)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, reads: torch.Tensor,
                stats: torch.Tensor) -> torch.Tensor:
        gru_output = self.gru_net(stats)
        attn_output = self.attn_net(reads)

        output = torch.cat([attn_output[:, 0], gru_output],
                           dim=-1)  # NxSx(E + 2H)
        return self.fc(output)

    def forward_train(self, reads: torch.Tensor, stats: torch.Tensor,
                      mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gru_output = self.gru_net(stats)
        attn_output = self.attn_net(reads)

        masked_output = self.mask_fc(attn_output[mask])

        output = torch.cat([attn_output[:, 0], gru_output],
                           dim=-1)  # NxSx(E + 2H)

        return self.fc(output), masked_output

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def read_masking(self,
                     reads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Masking
        # 1) MASK = we create our mask if > 0.8
        # 2) unkown -> we set elements which are unknown
        # 3) final mask -> bitwise-and (mask AND NOT unknown)
        # 4) we get values before_masking, but without strand info -> use for ground truth (target labels)
        # 5) We take random values 0-11 (with strand info) and apply it to the masked positions. after fc4 in forward it becomes 0-5
        mask = torch.rand(reads.size(),
                          device=self.device) < self.hparams.reads_mask_prob
        unknown = (reads == 5) | (reads == 11)
        mask &= ~unknown

        mask_target = torch.remainder(reads[mask], 6)

        reads[mask] = torch.randint(
            0,
            high=5,
            size=reads[mask].size(),
            dtype=torch.uint8,
            device=self.device) + torch.div(
                reads[mask], 6, rounding_mode='floor') * 6

        return mask_target, mask

    def training_step(self, batch, batch_idx):
        reads, labels, pos_stats = batch
        reads = reads.long()  # x.size() = B R S, y.size() = B S
        labels = labels.long()
        pos_stats = pos_stats.transpose(
            1, 2).float()  # B F S --> transpose --> B S F

        mask_target, mask = self.read_masking(reads)
        seq_logits, mask_logits = self.forward_train(reads, pos_stats, mask)
        seq_logits = seq_logits.transpose(1, 2)

        prediction_loss = F.cross_entropy(seq_logits, labels)
        self.log('prediction_loss', prediction_loss)

        masking_loss = F.cross_entropy(mask_logits, mask_target)
        self.log('masking_loss', masking_loss)

        total_loss = prediction_loss + self.hparams.alpha * masking_loss
        self.log('total_loss', total_loss)

        train_acc = self.train_accuracy(seq_logits, labels)
        self.log('train_acc', train_acc)

        return total_loss

    def validation_step(self, batch, batch_idx):
        reads, labels, pos_stats = batch
        reads = reads.long()  # x.size() = B R S, y.size() = B S
        labels = labels.long()
        pos_stats = pos_stats.transpose(
            1, 2).float()  # B F S --> transpose --> B S F

        seq_logits = self(reads, pos_stats)
        seq_logits = seq_logits.transpose(1, 2)

        loss = F.cross_entropy(seq_logits, labels)
        self.log('val_loss', loss, prog_bar=True)

        self.log('val_acc', self.val_accuracy(seq_logits, labels))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def get_trainer_defaults() -> Dict[str, Any]:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=-1,
        filename='{epoch}-{val_loss:.5f}-{val_acc_epoch:.5f}')

    trainer_defaults = {
        'callbacks': [checkpoint_callback],
        'logger': WandbLogger(project='docker_roko', log_model='all')
    }

    return trainer_defaults


def cli_main():
    LightningCLI(
        Polisher,
        RokoDataModule,
        seed_everything_default=
        42,  # Always put seed so we can repeat the experiment
        save_config_overwrite=True,
        trainer_defaults=get_trainer_defaults())


if __name__ == "__main__":
    cli_main()
