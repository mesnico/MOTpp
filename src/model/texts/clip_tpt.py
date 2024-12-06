from turtle import forward
import torch
import clip
from torch import nn

class CLIP(nn.Module):
    def __init__(self, 
                 latent_dim=256,
                 num_prompt_tokens=0,
                 vae=False) -> None:
        super().__init__()
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.clip_device, shallow_text_prompt_tokens=num_prompt_tokens)
        self.text_proj = nn.Linear(512, latent_dim)
        self.nfeats = latent_dim   # TODO: only needed for selecting the right encoder...
        self.vae = vae

    def forward(self, x_dict):
        text = x_dict["text"]
        text_tokenized = clip.tokenize(text, truncate=True).to(self.clip_device)
        
        text_features = self.clip_model.encode_text(text_tokenized).float()
        text_features = self.text_proj(text_features)
        if self.vae:
            text_features = text_features[:, 0:2]
        else:
            text_features = text_features[:, 0:1]

        return text_features