# ============================================================
# Pokemon Pair Regression - Training
# ============================================================

import os
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ""

TRAIN_CSV = "./train.csv"
MODEL_PATH = "./pokemon_resnet18_pair.pth"

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6

RANDOM_STATE = 42

NUM_WORKERS = 2


# ============================================================
# Reproducibility
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Transforms
# ============================================================

weights = ResNet18_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Mild augmentation only
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


val_transform = weights.transforms()


# ============================================================
# Dataset
# ============================================================

class PokemonDataset(Dataset):

    def __init__(self, df, data_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def get_image_path(self, path):
        path = path.replace("./images/", "")
        return os.path.join(self.data_dir, path)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img1_path = self.get_image_path(
            row["noisy_image1"]
        )

        img2_path = self.get_image_path(
            row["noisy_image2"]
        )

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        score = torch.tensor(
            row["result"],
            dtype=torch.float32
        )

        return img1, img2, score


# ============================================================
# Model
# ============================================================

class PokemonModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Pretrained ResNet18
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Remove classification layer
        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

        # We use:
        # |z1 - z2|  -> difference
        # z1 * z2    -> similarity
        #
        # Total = 1024 features

        self.regressor = nn.Sequential(

            nn.Linear(
                self.feature_dim * 2,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(
                64,
                1
            )
        )

    def encode(self, x):

        z = self.encoder(x)

        z = z.flatten(1)

        return z

    def forward(self, img1, img2):

        z1 = self.encode(img1)
        z2 = self.encode(img2)

        difference = torch.abs(z1 - z2)
        product = z1 * z2

        combined = torch.cat(
            [
                difference,
                product
            ],
            dim=1
        )

        score = self.regressor(combined)

        return score.squeeze(1)


# ============================================================
# Validation
# ============================================================

def validate(model, loader):

    model.eval()

    total_squared_error = 0.0
    total_examples = 0

    predictions_all = []
    targets_all = []

    with torch.no_grad():

        for img1, img2, scores in loader:

            img1 = img1.to(device)
            img2 = img2.to(device)
            scores = scores.to(device)

            predictions = model(
                img1,
                img2
            )

            squared_error = (
                predictions - scores
            ) ** 2

            total_squared_error += (
                squared_error.sum().item()
            )

            total_examples += len(scores)

            predictions_all.extend(
                predictions.cpu().numpy()
            )

            targets_all.extend(
                scores.cpu().numpy()
            )

    rmse = np.sqrt(
        total_squared_error /
        total_examples
    )

    return rmse


# ============================================================
# Training
# ============================================================

def train(model, train_loader, val_loader):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_rmse = float("inf")
    best_weights = None

    epochs_without_improvement = 0

    for epoch in range(EPOCHS):

        model.train()

        total_squared_error = 0.0
        total_examples = 0

        for img1, img2, scores in train_loader:

            img1 = img1.to(device)
            img2 = img2.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()

            predictions = model(
                img1,
                img2
            )

            loss = loss_fn(
                predictions,
                scores
            )

            loss.backward()

            optimizer.step()

            squared_error = (
                predictions - scores
            ) ** 2

            total_squared_error += (
                squared_error.sum().item()
            )

            total_examples += len(scores)

        train_rmse = np.sqrt(
            total_squared_error /
            total_examples
        )

        val_rmse = validate(
            model,
            val_loader
        )

        scheduler.step(val_rmse)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.7f}"
        )

        if val_rmse < best_rmse:

            best_rmse = val_rmse

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print(
                f"  -> New best model: "
                f"{best_rmse:.4f}"
            )

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print("Early stopping.")

            break

    if best_weights is not None:

        model.load_state_dict(
            best_weights
        )

    print()
    print(
        f"Best Validation RMSE: "
        f"{best_rmse:.4f}"
    )

    return model


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Pokemon Pair Regression")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(TRAIN_CSV)

    print(
        f"Total examples: {len(df)}"
    )

    print(
        "\nTarget statistics:"
    )

    print(
        df["result"].describe()
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print(
        f"\nTraining examples: {len(train_df)}"
    )

    print(
        f"Validation examples: {len(val_df)}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = PokemonDataset(
        train_df,
        DATA_DIR,
        transform=train_transform
    )

    val_dataset = PokemonDataset(
        val_df,
        DATA_DIR,
        transform=val_transform
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = PokemonModel()

    model = model.to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    model = train(
        model,
        train_loader,
        val_loader
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_rmse = validate(
        model,
        val_loader
    )

    print(
        f"\nFinal Validation RMSE: "
        f"{final_rmse:.4f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()