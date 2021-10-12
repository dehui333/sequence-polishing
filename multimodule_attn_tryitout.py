#train.py
import torch
from argparse import ArgumentParser
from progressbar import ProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
from THEDataModule import THEDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import math
from pytorch_lightning.loggers import WandbLogger
from evoformer import Evoformer


class THELightningModule(pl.LightningModule):

    def __init__(self, lr=1e-4, epochs=100, patience=30, in_size=500, hidden_size=128, num_layers=3, embedding_dim=256, heads=8, evoformer_blocks=8):
        super().__init__()
        #constructor

        # Model 
        self.embedding = nn.Embedding(12, embedding_dim)
        self.do = nn.Dropout(0.2)

        self.evoformer = Evoformer(embedding_dim, heads, evoformer_blocks)

        # self.fc1 = nn.Linear(200, 100)
        # self.do1 = nn.Dropout(0.2)

        # self.fc2 = nn.Linear(100, 10)
        # self.do2 = nn.Dropout(0.2)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_size = in_size

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

        #self.gru = nn.GRU(self.in_size, self.hidden_size, num_layers=self.num_layers,
        #                  batch_first=True, bidirectional=True, dropout=0.2)
        #stdv = math.sqrt(2.0 / self.gru.hidden_size)

        #for param in self.gru.parameters():
        #    if len(param.shape) >= 2:
        #        init.orthogonal_(param.data)
        #    else:
        #        init.normal_(param.data)

        self.fc4 = nn.Linear(embedding_dim, 5)

    def forward(self, x):
        #print("x input size: ", x.size()) # torch.Size([128, 200, 90]) B,R,S

        x = self.do(self.embedding(x))
        #print("x after embedding size: ", x.size()) # torch.Size([128, 200, 90, 64]) B,R,S,embd
        
        x = self.evoformer(x)
        #print("x after axial attention size: ", x.size()) # torch.Size([128, 200, 90, 64]) B,R,S,embd

        #x = x.permute((0, 2, 3, 1))
        #print("x after permute size: ", x.size()) # torch.Size([128, 90, 64, 200]) B,S,embd,R

        #print("x[:,:,:,0] size: ", x[:,:,:,0].size()) # torch.Size([128, 90, 64])

        return self.fc4(x[:,0]) # take the first row # torch.Size([B x S x F])

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.long()
        #print("x input size: ", x.size())
        #print("y input size: ", y.size())
        logits = self.forward(x).transpose(1, 2)
        #print("logits input size: ", logits.size())
        loss = self.cross_entropy_loss(logits, y)
        train_acc_batch = self.train_accuracy(logits, y)
        self.log('train_loss', loss)
        return loss


    def training_epoch_end(self, outputs):
        accuracy = self.train_accuracy.compute()
        self.log('Train_acc_epoch', accuracy, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.long()
        logits = self.forward(x).transpose(1, 2)
        loss = self.cross_entropy_loss(logits, y)
        val_acc_batch = self.val_accuracy(logits, y)
        self.log('val_acc_batch', val_acc_batch, prog_bar=True)
        self.log('val_loss_batch', loss, prog_bar=True)
        return loss


    def validation_epoch_end(self, outputs):
        accuracy = self.val_accuracy.compute()
        self.log('val_acc_epoch', accuracy, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main():
    # argument parser
    parser = ArgumentParser()
    parser.add_argument('datapath', type=str) # training set path or test set path
    parser.add_argument('out', type=str) # output file (for train) / directory (for test) path
    parser.add_argument('--valpath', type=str, default=None) # validation set path
    parser.add_argument('--memory', action='store_true', default=False)
    parser.add_argument('--t', type=int, default=0) # number of threads
    parser.add_argument('--b', type=int, default=16) # batch size
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--in_size", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--evoformer_blocks", type=int, default=8)

    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # weights and biases
    wandb_logger = WandbLogger(project='docker_roko', log_model='all')

    # early stopping
    early_stop_callback = EarlyStopping(monitor="val_acc_batch", patience=args.patience)

    # model checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_batch', dirpath=args.out, filename='sample-{val_acc_batch:.2f}')
    
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(args, gpus=[4,5,6,7], accelerator="ddp", logger=wandb_logger, callbacks=[early_stop_callback, checkpoint_callback])
    
    # data
    data = THEDataModule(args.datapath, args.b, args.memory, args.valpath, args.t)

    # Instantiate model
    model = THELightningModule(lr=args.lr, epochs=args.epochs, patience=args.patience, in_size=args.in_size, hidden_size=args.hidden_size, num_layers=args.num_layers, embedding_dim = args.embedding_dim, heads = args.heads, evoformer_blocks = args.evoformer_blocks)

    wandb_logger.watch(model)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()