import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from simple_cnn import SimpleCNN


# ===== Dataset Class =====
class SignDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = 1.0  # Binary label for presence of sign

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ===== Utilities =====
def load_all_training_annotations(training_root):
    all_rows = []
    for class_folder in os.listdir(training_root):
        class_dir = os.path.join(training_root, class_folder)
        if not os.path.isdir(class_dir):
            continue
        gt_csv = os.path.join(class_dir, f"GT-{class_folder}.csv")
        if os.path.isfile(gt_csv):
            df = pd.read_csv(gt_csv, sep=';')
            df["image_path"] = df["Filename"].apply(lambda x: os.path.join(class_dir, x))
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True)


# ===== Train/Validate Loop =====
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * inputs.size(0)
    acc = correct / len(dataloader.dataset)
    return total_loss / len(dataloader.dataset), acc


# ===== Main =====
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_img_root = os.path.join(base_dir, "data", "GTSRB", "Final_Training", "Images")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default=train_img_root)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_model", default="simple_cnn.pth")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # === Load data ===
    df = load_all_training_annotations(args.image_root)
    train_df, val_df = train_test_split(df, test_size=args.val_split, random_state=args.seed)

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    train_ds = SignDataset(train_df, transform)
    val_ds = SignDataset(val_df, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # === Model setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), args.output_model)
    print(f"\n✅ Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
