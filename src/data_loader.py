import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ScientificDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "allenai/scibert_scivocab_uncased"
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]

        # Process image
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # Tokenize text
        tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=32,
        )

        return image, tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()
