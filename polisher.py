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
import sys



class Polisher(pl.LightningModule):

    def __init__(self, model_name, lr=1e-4, epochs=100, patience=30, **model_parameters):
        super().__init__()
        #constructor
        self.model_name = model_name
        self.cls = getattr(Polisher,self.model_name)
        self.backbone_cls = self.cls(**model_parameters)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        # Hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

    class Original_roko(nn.Module):
        def __init__(self, **params):
            super().__init__()
            # Model 
            self.embedding = nn.Embedding(12, 50)
            self.do = nn.Dropout(0.2)

            self.fc1 = nn.Linear(128, 100)
            self.do1 = nn.Dropout(0.2)

            self.fc2 = nn.Linear(100, 10)
            self.do2 = nn.Dropout(0.2)

            self.hidden_size = params.get('hidden_size')

            self.num_layers = params.get('num_layers')
            self.in_size = params.get('in_size')

            self.gru = nn.GRU(self.in_size, self.hidden_size, num_layers=self.num_layers,
                          batch_first=True, bidirectional=True, dropout=0.2)
            stdv = math.sqrt(2.0 / self.gru.hidden_size)

            for param in self.gru.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

            self.fc4 = nn.Linear(2 * self.hidden_size, 5)

        def forward(self, x):
            x = self.do(self.embedding(x)) # B R S E
            x = x.permute((0, 2, 3, 1)) # B S E R

            x = F.relu(self.fc1(x)) # B S E 100
            x = self.do1(x)

            x = F.relu(self.fc2(x)) # B S E 10
            x = self.do2(x)

            x = x.reshape(-1, 128, self.in_size) # B S in_size=500
            x, _ = self.gru(x) # B S 2*hidden_size=256

            return self.fc4(x) # B S 5

    class Attention_roko(nn.Module):
        def __init__(self, **params):
            super().__init__()
            # Model 
            self.embedding = nn.Embedding(12, params.get('embedding_dim'))
            self.do = nn.Dropout(0.2)
            self.evoformer = Evoformer(msa_embedding_dim = params.get('embedding_dim'), heads = params.get('heads'), num_blocks = params.get('evoformer_blocks'))
            self.fc4 = nn.Linear(params.get('embedding_dim'), 5)
            self.mask_proj = nn.Linear(params.get('embedding_dim'), 5)

        def forward(self, x):
            x = self.do(self.embedding(x)) # B R S E
            x = self.evoformer(x) # B R S E
            # (B R S E) -> take the first row of R dimension -> (B S E) and pass it to linear layer
            return self.fc4(x[:,0]), self.mask_proj(x) # B S 5 and B R S 12



    def forward(self, x):
        return self.backbone_cls(x)


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
        
        # apply masks, x is still B R S
        x[rand] = torch.randint(0, high = 5, size=x[rand].size(), dtype=torch.uint8, device=self.device) + torch.div(x[rand], 6, rounding_mode='floor')*6

        if self.model_name == 'Attention_roko':
            logits, attn_out = self.forward(x) # logits = B C S (B x 5 x S), y = B S, attn_out = B R S 5
            masking_loss = self.cross_entropy_loss(attn_out[rand], before_masking)
        else:
            logits = self.forward(x)
            masking_loss = 0
        
        logits = logits.transpose(1,2)
        train_loss = self.cross_entropy_loss(logits, y)
        
        overall_loss = train_loss + 0.1*masking_loss # attn_out[rand].size() = number of elements masked x 5
        train_acc_batch = self.train_accuracy(logits, y)
        #self.log('train_loss', loss)
        self.log('overall_loss', overall_loss)
        self.log('train_loss', train_loss)
        self.log('masking_loss', masking_loss)
        sys.exit()
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
    parser.add_argument('backbone', type=str) # choose attn or rnn
    parser.add_argument('--valpath', type=str, default=None) # validation set path
    parser.add_argument('--memory', action='store_true', default=False)
    parser.add_argument('--t', type=int, default=16) # number of threads
    parser.add_argument('--b', type=int, default=16) # batch size

    # hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)

    # roko_attn parameters
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--evoformer_blocks", type=int, default=8)

    # roko_rnn parameters
    parser.add_argument("--in_size", type=int, default=500)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)


    
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # weights and biases
    wandb_logger = WandbLogger(project='docker_roko', log_model='all')

    # early stopping
    early_stop_callback = EarlyStopping(monitor="val_acc_batch", patience=args.patience)

    # model checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_batch', dirpath=args.out, filename='sample-{val_acc_batch:.2f}')
    
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(args, gpus=[0,1,3], strategy="ddp", gradient_clip_val=1.0, logger=wandb_logger, callbacks=[early_stop_callback, checkpoint_callback]) #track_grad_norm=2, limit_train_batches=100, limit_val_batches=100)
    
    # data
    data = THEDataModule(args.datapath, args.b, args.memory, args.valpath, args.t)

    # Instantiate model
    model = Polisher(model_name = args.backbone, lr=args.lr, epochs=args.epochs, patience=args.patience, embedding_dim = args.embedding_dim, heads = args.heads, evoformer_blocks = args.evoformer_blocks, hidden_size = args.hidden_size, in_size = args.in_size, num_layers = args.num_layers)

    wandb_logger.watch(model)

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()