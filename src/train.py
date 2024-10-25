import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import ScientificDataset
from src.model import CLIPScientificModel


def contrastive_loss(image_features, text_features, temperature=0.07):
    logits = torch.matmul(image_features, text_features.T) / temperature
    labels = torch.arange(logits.shape[0]).to(logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_i + loss_t) / 2


def train_model():
    # Dataset and DataLoader
    dataset = ScientificDataset(csv_file="data/dataset.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model
    model = CLIPScientificModel().cuda()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    for epoch in range(10):  # Set appropriate number of epochs
        epoch_loss = 0.0
        for images, input_ids, attention_mask in dataloader:
            images, input_ids, attention_mask = (
                images.cuda(),
                input_ids.cuda(),
                attention_mask.cuda(),
            )

            # Forward pass
            image_features, text_features = model(images, input_ids, attention_mask)

            # Compute loss
            loss = contrastive_loss(image_features, text_features)
            epoch_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}], Loss: {epoch_loss / len(dataloader):.4f}")
