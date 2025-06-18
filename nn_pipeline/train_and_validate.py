# nn_pipeline/train_and_validate.py

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
import plotly.graph_objects as go


class SignDataset(Dataset):
    """
    Custom dataset for loading traffic sign images and their bounding boxes.
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        x = row["Roi.X1"]
        y = row["Roi.Y1"]
        w = row["Roi.X2"] - row["Roi.X1"]
        h = row["Roi.Y2"] - row["Roi.Y1"]
        target = torch.tensor([x, y, w, h], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, target


def load_all_training_annotations(training_root: str) -> pd.DataFrame:
    """
    Loads all ground truth CSV files from GTSRB training directory.

    Args:
        training_root (str): Root path of training image folders.

    Returns:
        pd.DataFrame: Combined dataframe of all annotations.
    """
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


def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def validate(model: nn.Module, dataloader: DataLoader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def plot_losses(loss_log: list[dict], output_html_path: str) -> None:
    """
    Save an interactive loss curve plot as HTML.

    Args:
        loss_log (list[dict]): List of epoch loss entries.
        output_html_path (str): Path to output HTML file.
    """
    df = pd.DataFrame(loss_log)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["train_loss"], mode="lines+markers", name="Train Loss"))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"], mode="lines+markers", name="Validation Loss"))
    fig.update_layout(
        title="Training vs Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white"
    )
    fig.write_html(output_html_path)
    print(f"📊 Loss plot saved to: {output_html_path}")


def main() -> None:
    """
    Entry point: Trains the SimpleCNN on the GTSRB dataset for bounding box regression.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_img_root = os.path.join(base_dir, "data", "GTSRB", "Final_Training", "Images")

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default=train_img_root)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_model", default="simple_cnn.pth")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_outputs=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    loss_log = []
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        loss_log.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

    # Save model
    torch.save(model.state_dict(), args.output_model)
    print(f"\n✅ Model saved to {args.output_model}")

    # Save loss log
    loss_csv_path = os.path.splitext(args.output_model)[0] + "_losses.csv"
    pd.DataFrame(loss_log).to_csv(loss_csv_path, index=False)
    print(f"📈 Saved loss log to: {loss_csv_path}")

    # Plot loss
    loss_plot_path = os.path.splitext(args.output_model)[0] + "_loss_plot.html"
    plot_losses(loss_log, loss_plot_path)


if __name__ == "__main__":
    main()
