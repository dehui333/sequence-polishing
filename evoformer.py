import torch
from torch import nn
from torch import functional as F
from attentions import MSAGatedAttention

# stochastic depth linear decay
def depth_prob(p_lowest: float, layer: int, n_layers: int) -> float:
    #print(layer, p_lowest, n_layers)
    return 1 - layer / n_layers * (1 - p_lowest)

class Evoformer(nn.Module):
    def __init__(self, msa_embedding_dim, heads, num_blocks, p_keep_lowest):
        super().__init__()

        prob_fn = lambda i: depth_prob(p_keep_lowest, i, num_blocks)
        self.blocks = nn.Sequential(*[EvoformerBlock(msa_embedding_dim, heads, prob_fn(i+1)) for i in range(num_blocks)])

    def forward(self, msa_repr):
        return self.blocks(msa_repr)


class EvoformerBlock(nn.Module):
    def __init__(self, msa_embedding_dim, heads, p_keep):
        super().__init__()

        self.msa_row_att = MSAGatedAttention('row', msa_embedding_dim, heads)
        self.msa_col_att = MSAGatedAttention('column', msa_embedding_dim, heads)
        self.p_keep = p_keep
        self.msa_transition = Transition(msa_embedding_dim, projection_factor=4)

    def forward(self, msa_repr):
        if not self.training or torch.rand(1) <= self.p_keep:
            #print("evo")
            # MSA track
            msa_repr = msa_repr + self.msa_row_att(msa_repr)
            msa_repr = msa_repr + self.msa_col_att(msa_repr)
            msa_repr = msa_repr + self.msa_transition(msa_repr)
            return msa_repr
        else:
            print("skipped")
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
