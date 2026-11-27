import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# Model (must match training architecture exactly)
# ============================================================

class PokemonModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

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
        z1 = self.encoder(img1)
        z2 = self.encoder(img2)

        difference = torch.abs(z1 - z2)
        product = z1 * z2

        combined = torch.cat(
            [z1, z2, difference, product],
            dim=1
        )

        score = self.regressor(combined)

        return score.squeeze(-1)


# ============================================================
# Preprocessing (same as validation)
# ============================================================

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])


def preprocess(img_path):
    with Image.open(img_path) as img:
        img = img.convert("L")
        return test_transform(img)


# ============================================================
# Test Dataset
# ============================================================

class PokemonTestDataset(Dataset):

    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1_path = os.path.join(
            self.data_dir,
            row["noisy_image1"].replace("./images/", "")
        )

        img2_path = os.path.join(
            self.data_dir,
            row["noisy_image2"].replace("./images/", "")
        )

        return preprocess(img1_path), preprocess(img2_path)


# ============================================================
# Inference and Submission
# ============================================================

def create_submission(
    model_path="./pokemon_model_2.pth",
    test_csv="./test.csv",
    sample_csv="./sample_submission.csv",
    data_dir="",
    output_csv="./submission.csv"
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load trained model
    model = PokemonModel().to(device)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully!")

    # Load test data
    test_df = pd.read_csv(test_csv)
    sample_df = pd.read_csv(sample_csv)

    assert len(test_df) == 80000, "Unexpected test size"
    assert len(sample_df) == 80000, "Unexpected sample size"

    test_dataset = PokemonTestDataset(test_df, data_dir)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # Generate predictions
    predictions = []

    with torch.inference_mode():
        for batch_idx, (img1, img2) in enumerate(test_loader):
            img1 = img1.to(device)
            img2 = img2.to(device)

            preds = model(img1, img2)

            predictions.extend(
                preds.cpu().numpy().reshape(-1)
            )

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Processed {len(predictions):,}"
                    f"/{len(test_df):,} examples"
                )

    # Convert and clip predictions
    predictions = np.asarray(predictions)

    assert np.isfinite(predictions).all(), \
        "Predictions contain NaN or infinity"

    predictions = np.clip(predictions, 0.0, 10.0)

    assert len(predictions) == len(sample_df)

    # Preserve sample submission row IDs and order
    if "row_id" in test_df.columns:
        results = pd.DataFrame({
            "row_id": test_df["row_id"],
            "result": predictions
        })

        submission = sample_df[["row_id"]].merge(
            results,
            on="row_id",
            how="left",
            sort=False,
            validate="one_to_one"
        )
    else:
        # Assumes test.csv follows sample_submission.csv order
        submission = pd.DataFrame({
            "row_id": sample_df["row_id"],
            "result": predictions
        })

    # Verify submission
    assert len(submission) == 80000
    assert submission["row_id"].is_unique
    assert submission["result"].notna().all()
    assert submission["result"].between(0, 10).all()
    assert submission["row_id"].equals(sample_df["row_id"])

    # Save submission
    submission.to_csv(output_csv, index=False)

    print(f"\nSaved {output_csv} with {len(submission)} rows.")
    print(submission.head())

    print("\nPrediction statistics:")
    print(submission["result"].describe())


if __name__ == "__main__":
    create_submission()
