import timm
import torch
import torch.nn as nn
from transformers import AutoModel


class CLIPScientificModel(nn.Module):
    def __init__(self):
        super(CLIPScientificModel, self).__init__()
        # Load SciBERT as the text encoder
        self.text_encoder = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        # Load Vision Transformer (ViT) as the image encoder
        self.image_encoder = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.image_encoder.head = nn.Linear(self.image_encoder.head.in_features, 768)

    def forward(self, image, input_ids, attention_mask):
        # Image encoding
        image_features = self.image_encoder(image)

        # Text encoding
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]

        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)

        return image_features, text_features
