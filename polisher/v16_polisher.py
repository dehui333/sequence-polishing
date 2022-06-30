#train.py
from cProfile import label
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchmetrics
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.cli import LightningCLI

#from v6v7v8v9v10v11v12v13_evoformer import Evoformer, PositionalEncoding
from v13_roko_data_module import RokoDataModule
#import sys

POSITIONAL_FEATURES = 5
COV_FEATURES = 1
READ_FEATURES = 12
OUTPUT_CLASSES = 5
MASK_CLASSES = 5
SEQ_LEN = 90
NUM_R = 30

class GRUNetwork(nn.Module):
    def __init__(self,
                 input_dim: int, # P
                 hidden_size: int,
                 n_layers: int,
                 dropout: float = 0.0,
                 mode: str = 'pos_stat') -> None:
        super().__init__()

        # pos_stat network or cov network
        self.linear_in_dim = POSITIONAL_FEATURES if mode == 'pos_stat' else COV_FEATURES
        self.linear = nn.Linear(self.linear_in_dim, input_dim) # F P
        self.gru = nn.GRU(input_dim, 
                          hidden_size, 
                          num_layers = n_layers,
                          dropout=dropout,
                          batch_first=True, 
                          bidirectional=True)

    def forward(self, pos_stat: torch.Tensor) -> torch.Tensor:
        pos_stat = F.relu(self.linear(pos_stat)) # should be B S P
        pos_stat, _ = self.gru(pos_stat) # B S 256=2*hidden_size
        return pos_stat # BxNxSx2H


class Polisher(pl.LightningModule):

    def __init__(self, 
                 gru_input_dim: int = 128,
                 gru_hidden_dim: int =128,
                 gru_n_layers: int = 1,
                 gru_dropout: float = 0.0,
                 lr=3e-4) -> None:
        super().__init__()
        #constructor
        self.save_hyperparameters()

        self.gru = GRUNetwork(gru_input_dim, gru_hidden_dim, gru_n_layers,
                              gru_dropout, mode = 'pos_stat')
        self.gru2 = GRUNetwork(gru_input_dim, gru_hidden_dim, gru_n_layers,
                              gru_dropout, mode = 'cov')
        self.fc = nn.Linear(2 * gru_hidden_dim + 2 * gru_hidden_dim, OUTPUT_CLASSES)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, 
                pos_stat: torch.Tensor,
                cov: torch.Tensor) -> torch.Tensor: # pos_stat B S F, cov B S 1
        
        gru_output = self.gru(pos_stat) # B S 2H
        gru_output2 = self.gru2(cov) # B S 2H
        output = torch.cat((gru_output, gru_output2), 2) # do not use attn
        return self.fc(output)

    def forward_train(self,
                      pos_stat: torch.Tensor,
                      cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # pos_stat B S F, cov B S 1
        
        gru_output = self.gru(pos_stat) # B S 2H
        gru_output2 = self.gru2(cov) # B S 2H

        output = torch.cat((gru_output, gru_output2), 2)

        return self.fc(output) # B S 5 and N_masked 5


    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        _, labels, pos_stat, cov = batch
        labels = labels.long()
        pos_stat = pos_stat.transpose(1,2).float() # B F S --> transpose --> B S F
        cov = cov.transpose(1,2).float() # B 1 S --> transpose --> B S 1
        seq_logits = self.forward_train(pos_stat, cov)

        seq_logits = seq_logits.transpose(1,2)
        prediction_loss = self.cross_entropy_loss(seq_logits, labels)
        self.log('prediction_loss', prediction_loss)


        train_acc_batch = self.train_accuracy(seq_logits, labels)
        self.log('train_acc', train_acc_batch)
        #sys.exit()
        
        return prediction_loss

    def validation_step(self, batch, batch_idx):
        _, labels, pos_stat, cov = batch
        labels = labels.long()
        pos_stat = pos_stat.transpose(1,2).float()
        cov = cov.transpose(1,2).float()
        
        seq_logits = self.forward(pos_stat, cov)
        seq_logits = seq_logits.transpose(1,2) # logits = B C S
        loss = self.cross_entropy_loss(seq_logits, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy(seq_logits, labels))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

def get_trainer_defaults() -> Dict[str, Any]:
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=-1,
        filename='{epoch}-{val_loss:.5f}-{val_acc:.5f}')

    trainer_defaults = {
        'callbacks': [checkpoint_callback],
        'logger': WandbLogger(project='docker_roko', log_model='all', save_dir=os.getcwd()), # weights and biases
        'strategy': DDPPlugin(find_unused_parameters = True) # 'strategy' parameter requires pytorch lightning v1.5 and above
    }

    return trainer_defaults

def cli_main():
    LightningCLI(
        Polisher,
        RokoDataModule,
        seed_everything_default = 42, # always put seed so we can repeat the experiment
        save_config_overwrite=True,
        trainer_defaults=get_trainer_defaults())

if __name__ == "__main__":
    cli_main()
