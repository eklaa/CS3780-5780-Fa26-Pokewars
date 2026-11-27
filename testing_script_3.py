# ============================================================
# Pokemon Pair Regression - Testing
# ============================================================

import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# Configuration
# ============================================================

DATA_DIR = ""

TEST_CSV = "./test.csv"

MODEL_PATH = "./pokemon_resnet18_pair.pth"

OUTPUT_CSV = "./predictions.csv"

BATCH_SIZE = 64

NUM_WORKERS = 2


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Transform
# ============================================================

weights = ResNet18_Weights.DEFAULT

test_transform = weights.transforms()


# ============================================================
# Dataset
# ============================================================

class PokemonTestDataset(Dataset):

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

    def get_image_path(self, path):

        path = path.replace(
            "./images/",
            ""
        )

        return os.path.join(
            self.data_dir,
            path
        )

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img1_path = self.get_image_path(
            row["noisy_image1"]
        )

        img2_path = self.get_image_path(
            row["noisy_image2"]
        )

        img1 = Image.open(
            img1_path
        ).convert("RGB")

        img2 = Image.open(
            img2_path
        ).convert("RGB")

        if self.transform is not None:

            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


# ============================================================
# Model
# ============================================================

class PokemonModel(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = resnet18(
            weights=None
        )

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.feature_dim = 512

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

        difference = torch.abs(
            z1 - z2
        )

        product = z1 * z2

        combined = torch.cat(
            [
                difference,
                product
            ],
            dim=1
        )

        score = self.regressor(
            combined
        )

        return score.squeeze(1)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Pokemon Model Testing")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_CSV
    )

    print(
        f"Test examples: {len(test_df)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = PokemonTestDataset(
        test_df,
        DATA_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = PokemonModel()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    print(
        f"Loaded model from: "
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = []

    with torch.no_grad():

        for img1, img2 in test_loader:

            img1 = img1.to(device)
            img2 = img2.to(device)

            scores = model(
                img1,
                img2
            )

            predictions.extend(
                scores.cpu().numpy()
            )

    predictions = np.array(
        predictions
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    output_df = test_df.copy()

    output_df["result"] = predictions

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\nSaved predictions to: "
        f"{OUTPUT_CSV}"
    )

    print(
        "\nPrediction statistics:"
    )

    print(
        pd.Series(predictions).describe()
    )


if __name__ == "__main__":
    main()