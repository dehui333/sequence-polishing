import torch
from torch import nn
from torch import functional as F
from attentions import MSAGatedAttention

class Evoformer(nn.Module):
    def __init__(self, msa_embedding_dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.Sequential(*[EvoformerBlock(msa_embedding_dim, heads) for i in range(num_blocks)])

    def forward(self, msa_repr):
        return self.blocks(msa_repr)


class EvoformerBlock(nn.Module):
    def __init__(self, msa_embedding_dim, heads):
        super().__init__()

        self.msa_row_att = MSAGatedAttention('row', msa_embedding_dim, heads)
        self.msa_col_att = MSAGatedAttention('column', msa_embedding_dim, heads)

        self.msa_transition = Transition(msa_embedding_dim, projection_factor=4)

    def forward(self, msa_repr):
        #TODO add dropouts
        # MSA track
        msa_repr = msa_repr + self.msa_row_att(msa_repr)
        msa_repr = msa_repr + self.msa_col_att(msa_repr)
        msa_repr = msa_repr + self.msa_transition(msa_repr)

        return msa_repr

class Transition(nn.Module):
    def __init__(self, embedding_dim, projection_factor=4):
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, projection_factor * embedding_dim)
        self.linear2 = nn.Linear(projection_factor * embedding_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.linear2(torch.relu(x))
        return x
