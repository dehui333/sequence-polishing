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

    def __init__(self, lr=1e-4, epochs=100, patience=30, embedding_dim=128, heads=8, evoformer_blocks=8):
        super().__init__()
        #constructor

        # Model 
        self.embedding = nn.Embedding(12, embedding_dim)
        self.do = nn.Dropout(0.2)

        self.evoformer = Evoformer(embedding_dim, heads, evoformer_blocks)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

        self.fc4 = nn.Linear(embedding_dim, 5)
        self.mask_proj = nn.Linear(embedding_dim, 5)

    def forward(self, x):
        #print("x input size: ", x.size()) # torch.Size([128, 200, 90]) B,R,S

        x = self.do(self.embedding(x))
        #print("x after embedding size: ", x.size()) # torch.Size([128, 200, 90, 64]) B,R,S,embd
        
        x = self.evoformer(x)
        #print("x after axial attention size: ", x.size()) # torch.Size([128, 200, 90, 64]) B,R,S,embd

        # take the first row and pass it to linear layer torch.Size([B x S x F])
        return self.fc4(x[:,0]), self.mask_proj(x) # returning B x S x 5 and B x R x S x 12


    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.long() # x.size() = B R S, y.size() = B S
        
        # generate a boolean mask
        probs = torch.rand(x.size(), device = self.device) > 0.85 # a tensor with true and false values

        # a boolean tensor with true at positions where the values are 5 or 11
        non_unknown = (x == 5) + (x == 11) # true if x at the position is 5 or 11

        # for each position, if negation of non_unknown is true (not 5 not 11), then use probs, otherwise use negation of non_known (false at that position if x at that position is 5 or 11
        rand = torch.where(~non_unknown > 0, probs, ~non_unknown)  

        # save original values
        before_masking = torch.remainder(x[rand],6).clone() # size = number of elements masked (1 dimensional)
        
        # apply masks
        x[rand] = torch.randint(0, high = 5, size=x[rand].size(), dtype=torch.uint8, device=self.device) + torch.div(x[rand], 6, rounding_mode='floor')*6

        logits, attn_out = self.forward(x) # logits = B C S (B x 5 x S), y = B S, attn_out = B R S 5
        logits = logits.transpose(1,2)
        train_loss = self.cross_entropy_loss(logits, y)
        masking_loss = self.cross_entropy_loss(attn_out[rand], before_masking)
        overall_loss = train_loss + 0.1*masking_loss # attn_out[rand].size() = number of elements masked x 5
        train_acc_batch = self.train_accuracy(logits, y)
        #self.log('train_loss', loss)
        self.log('overall_loss', overall_loss)
        self.log('train_loss', train_loss)
        self.log('masking_loss', masking_loss)
        return overall_loss


    def training_epoch_end(self, outputs):
        accuracy = self.train_accuracy.compute()
        self.log('Train_acc_epoch', accuracy, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.long()
        logits, _ = self.forward(x)
        logits = logits.transpose(1,2) # logits = B C S
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
    parser.add_argument('--t', type=int, default=16) # number of threads
    parser.add_argument('--b', type=int, default=16) # batch size
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--embedding_dim", type=int, default=128)
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
    trainer = pl.Trainer.from_argparse_args(args, gpus=[2,3,4,5], strategy="ddp", gradient_clip_val=1.0, logger=wandb_logger, callbacks=[early_stop_callback, checkpoint_callback]) #track_grad_norm=2, limit_train_batches=100, limit_val_batches=100)
    
    # data
    data = THEDataModule(args.datapath, args.b, args.memory, args.valpath, args.t)

    # Instantiate model
    model = THELightningModule(lr=args.lr, epochs=args.epochs, patience=args.patience, embedding_dim = args.embedding_dim, heads = args.heads, evoformer_blocks = args.evoformer_blocks)

    wandb_logger.watch(model)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()