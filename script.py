# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import copy

class PokemonModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, img1, img2):
        z1 = self.encoder(img1)
        z2 = self.encoder(img2)

        combined = torch.cat([z1, z2], dim=1)

        score = self.regressor(combined)

        return score.squeeze(-1)

class PokemonDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
    
        img1_path = os.path.join(
            self.data_dir, row["noisy_image1"].replace("./images/", "")
        )
    
        img2_path = os.path.join(
            self.data_dir, row["noisy_image2"].replace("./images/", "")
        )

        img1 = preprocess(img1_path)
        img2 = preprocess(img2_path)
    
        score = torch.tensor(
            row["result"],
            dtype=torch.float32
        )
    
        return img1, img2, score

def split(ds_path):
    df = pd.read_csv(ds_path)

    train_ds, val_ds = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    return train_ds, val_ds

def preprocess(img_path):
    img = Image.open(img_path).convert("L")

    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = torch.from_numpy(img).unsqueeze(0)

    return img

def build_model():
    return PokemonModel()

def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epochs = 20
    best_rmse = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for img1, img2, scores in train_loader:
            predictions = model(img1, img2)
            loss = loss_fn(predictions, scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(scores)

        train_rmse = np.sqrt(total_loss / len(train_loader.dataset))
        val_rmse = validate(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"Train RMSE = {train_rmse:.4f}, "
            f"Val RMSE = {val_rmse:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    print(f"Best Validation RMSE: {best_rmse:.4f}")

    return model

def validate(model, val_loader):
    model.eval()

    total_squared_error = 0
    total_examples = 0

    with torch.no_grad():
        for img1, img2, scores in val_loader:
            predictions = model(img1, img2)

            total_squared_error += (
                (predictions - scores) ** 2
            ).sum().item()

            total_examples += len(scores)

    rmse = np.sqrt(total_squared_error / total_examples)

    return rmse

def main():
    DATA_DIR = ""
    DATA_PATH ="./train.csv"

    print(DATA_DIR+DATA_PATH)

    train_df, val_df = split(DATA_PATH)

    train_dataset = PokemonDataset(train_df, DATA_DIR)
    val_dataset = PokemonDataset(val_df, DATA_DIR)

    print("SPLIT COMPLETE")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    model = build_model()

    print("MODEL BUILT")

    model = train(model, train_loader, val_loader)

    print("MODEL TRAINED")
    
    rmse = validate(model, val_loader)
    
    print("Validation RMSE:", rmse)

    return model

if __name__=="__main__":
    model = main()
    torch.save(model.state_dict(), "./pokemon_model.pth")