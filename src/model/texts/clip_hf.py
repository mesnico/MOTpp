from torch import nn
from transformers import CLIPModel, CLIPTokenizer, CLIPConfig
import torch

class CLIPHF(nn.Module):
    def __init__(self, latent_dim, clip_model_name="openai/clip-vit-base-patch32", vae=False):
        super(CLIPHF, self).__init__()
        self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the CLIP model and processor
        # config = CLIPConfig.from_pretrained(clip_model_name, output_hidden_states=True)
        clip_model = CLIPModel.from_pretrained(clip_model_name, output_hidden_states=True)
        self.text_model = clip_model.text_model
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Extract the last layer output dimension from the CLIP model
        last_layer_dim = clip_model.config.projection_dim
        
        # Define the transformer encoder layer for further processing
        transformer_encoder = nn.TransformerEncoderLayer(d_model=last_layer_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder, num_layers=2)
        
        # Linear layer for extracting the CLS feature
        self.text_proj = nn.Linear(last_layer_dim, latent_dim)
        self.vae = vae
        self.nfeats = latent_dim
        
    def forward(self, x_dict):
        input_text = x_dict["text"]

        # Get CLIP model outputs
        with torch.no_grad():
            inputs = self.tokenizer(text=input_text, return_tensors="pt", padding=True, max_length=77, truncation=True).to(self.clip_device)
            clip_outputs = self.text_model(**inputs)

        # Extract last layer output
        last_layer_output = clip_outputs.last_hidden_state
        
        # Process using transformer encoder
        attention_mask = ~inputs["attention_mask"].bool()
        processed_output = self.transformer_encoder(last_layer_output, src_key_padding_mask=attention_mask)

        # Extract CLS feature
        cls_token = self.text_proj(processed_output[:, :2, :])
        if not self.vae:
            cls_token = cls_token[:, 0, :]
        
        return cls_token