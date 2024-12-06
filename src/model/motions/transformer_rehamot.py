from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, Tensor

def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self,
                 nfeats: int,
                 latent_dim: int = 256,
                 embed_size: int = 1024,
                 ff_size: int = 1024,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 only_return_cls_token: bool = True,
                 activation: str = "gelu",
                 vae: bool = False,
                 **kwargs) -> None:
        super(TransformerEncoder, self).__init__()

        input_feats = nfeats
        self.nfeats = nfeats
        self.only_return_cls_token = only_return_cls_token
        self.skel_embedding = nn.Linear(input_feats, latent_dim)
        self.vae = vae

        self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

        self.fc = nn.Linear(latent_dim, embed_size)

        self.learning_rates_x = []

    def forward(self, x_dict: dict):
        features = x_dict["x"]
        lengths = x_dict["length"]

        device = features.device

        bs, nframes, nfeats = features.shape
        mask = lengths_to_mask(lengths, device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        # Each batch has its own set of tokens
        emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

        # adding the embedding token for all sequences
        xseq = torch.cat((emb_token[None], x), 0)

        # create a bigger mask, to allow attend to emb
        token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
        aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        features = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)
        if self.only_return_cls_token:
            # normalization in the motion embedding space
            features = F.normalize(features[0], dim=1)
            # normalization in the joint embedding space
            return F.normalize(self.fc(features), dim=1), None
        elif self.vae:
            features = features.permute(1, 0, 2)
            return features[:, :2, :]
        else:
            # return none-norm sequence features
            features = self.fc(features).permute(1,0,2)
            features = F.normalize(features, dim=-1)
            return features, aug_mask
