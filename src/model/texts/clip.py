from turtle import forward
import torch
import clip
from torch import nn

class CLIP(nn.Module):
    def __init__(self, 
                 latent_dim=256,
                 vae=False) -> None:
        super().__init__()
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device)
        self.text_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_dim * 2 if vae else latent_dim)
        )
        self.nfeats = latent_dim   # TODO: only needed for selecting the right encoder...
        self.vae = vae

    def forward(self, x_dict):
        text = x_dict["text"]
        text_tokenized = clip.tokenize(text, truncate=True).to(self.clip_device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokenized).float()
        text_features = self.text_proj(text_features)
        if self.vae:
            text_features = text_features.view(-1, 2, self.nfeats)
        else:
            text_features = text_features.view(-1, 1, self.nfeats)

        return text_features