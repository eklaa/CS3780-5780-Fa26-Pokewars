# ============================================================
# Pokemon Image Pair Regression Model
# ============================================================

import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ""
DATA_PATH = "./train.csv"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

RANDOM_STATE = 42

MODEL_PATH = "./pokemon_model_2.pth"


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
# Image Transforms
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),

    transforms.ToTensor(),

    # Convert [0, 1] -> approximately [-1, 1]
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])


val_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])


# ============================================================
# Model
# ============================================================

class PokemonModel(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Shared CNN encoder
        # ----------------------------------------------------

        self.encoder = nn.Sequential(

            # Input: 1 x H x W
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 32 -> 64
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 64 -> 128
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 128 -> 256
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Output becomes 256-dimensional vector
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # ----------------------------------------------------
        # Pairwise regressor
        #
        # z1                  = image 1 features
        # z2                  = image 2 features
        # abs(z1 - z2)        = difference
        # z1 * z2             = similarity information
        #
        # 256 * 4 = 1024 total features
        # ----------------------------------------------------

        self.regressor = nn.Sequential(

            nn.Linear(256 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, img1, img2):

        # Encode both images using the SAME encoder
        z1 = self.encoder(img1)
        z2 = self.encoder(img2)

        # Difference between images
        difference = torch.abs(z1 - z2)

        # Element-wise similarity
        product = z1 * z2

        # Combine all information
        combined = torch.cat(
            [
                z1,
                z2,
                difference,
                product
            ],
            dim=1
        )

        score = self.regressor(combined)

        return score.squeeze(-1)


# ============================================================
# Dataset
# ============================================================

class PokemonDataset(Dataset):

    def __init__(
        self,
        df,
        data_dir,
        transform=None
    ):

        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ----------------------------------------------------
        # Image 1 path
        # ----------------------------------------------------

        img1_path = os.path.join(
            self.data_dir,
            row["noisy_image1"].replace(
                "./images/",
                ""
            )
        )

        # ----------------------------------------------------
        # Image 2 path
        # ----------------------------------------------------

        img2_path = os.path.join(
            self.data_dir,
            row["noisy_image2"].replace(
                "./images/",
                ""
            )
        )

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

        img1 = Image.open(img1_path).convert("L")
        img2 = Image.open(img2_path).convert("L")

        # ----------------------------------------------------
        # Apply transformations
        # ----------------------------------------------------

        if self.transform is not None:

            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        score = torch.tensor(
            row["result"],
            dtype=torch.float32
        )

        return img1, img2, score


# ============================================================
# Train / Validation Split
# ============================================================

def split_dataset(ds_path):

    df = pd.read_csv(ds_path)

    print("Total examples:", len(df))

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    print("Training examples:", len(train_df))
    print("Validation examples:", len(val_df))

    return train_df, val_df


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    val_loader
):

    model.eval()

    total_squared_error = 0.0
    total_examples = 0

    with torch.no_grad():

        for img1, img2, scores in val_loader:

            # Move data to GPU/CPU
            img1 = img1.to(device)
            img2 = img2.to(device)
            scores = scores.to(device)

            # Predictions
            predictions = model(
                img1,
                img2
            )

            # Squared error
            squared_error = (
                (predictions - scores) ** 2
            )

            total_squared_error += (
                squared_error.sum().item()
            )

            total_examples += len(scores)

    # RMSE
    rmse = np.sqrt(
        total_squared_error /
        total_examples
    )

    return rmse


# ============================================================
# Training
# ============================================================

def train(
    model,
    train_loader,
    val_loader
):

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = nn.MSELoss()

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------

    best_rmse = float("inf")
    best_weights = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_squared_error = 0.0
        total_examples = 0

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for img1, img2, scores in train_loader:

            # Move to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            scores = scores.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(
                img1,
                img2
            )

            # Calculate loss
            loss = loss_fn(
                predictions,
                scores
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            # Track RMSE
            squared_error = (
                (predictions - scores) ** 2
            )

            total_squared_error += (
                squared_error.sum().item()
            )

            total_examples += len(scores)

        # ----------------------------------------------------
        # Training RMSE
        # ----------------------------------------------------

        train_rmse = np.sqrt(
            total_squared_error /
            total_examples
        )

        # ----------------------------------------------------
        # Validation RMSE
        # ----------------------------------------------------

        val_rmse = validate(
            model,
            val_loader
        )

        # ----------------------------------------------------
        # Update learning rate
        # ----------------------------------------------------

        scheduler.step(val_rmse)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train RMSE: {train_rmse:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_rmse < best_rmse:

            best_rmse = val_rmse

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print(
                f"  -> New best model! "
                f"RMSE = {best_rmse:.4f}"
            )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    # --------------------------------------------------------
    # Restore best weights
    # --------------------------------------------------------

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
    print("Pokemon Model Training")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    train_df, val_df = split_dataset(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Create datasets
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

    print()
    print("Datasets created.")

    # --------------------------------------------------------
    # Create data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    print("DataLoaders created.")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = PokemonModel()

    model = model.to(device)

    print()
    print("Model built.")

    # Number of parameters
    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("Starting training...")
    print()

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

    print()
    print("=" * 60)
    print(
        f"Final Validation RMSE: "
        f"{final_rmse:.4f}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print(
        f"Model saved to: {MODEL_PATH}"
    )

    return model


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    model = main()