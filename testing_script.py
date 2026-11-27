import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from script import PokemonDataset, PokemonModel

def preprocess(img_path):
    img = Image.open(img_path).convert("L")

    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = torch.from_numpy(img).unsqueeze(0)

    return img

# 2. Test Dataset (preserves exact row order)
class PokemonTestDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1_path = os.path.join(self.data_dir, row["noisy_image1"].replace("./images/", ""))
        img2_path = os.path.join(self.data_dir, row["noisy_image2"].replace("./images/", ""))
        return preprocess(img1_path), preprocess(img2_path)

# 3. Inference & Submission Generation
def create_submission(model_path="./pokemon_model_2.pth", test_csv="./test.csv", data_dir=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model weights
    model = PokemonModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load test data (shuffle=False is critical to keep row_id aligned)
    test_df = pd.read_csv(test_csv)
    test_dataset = PokemonTestDataset(test_df, data_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for img1, img2 in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            preds = model(img1, img2)
            predictions.extend(preds.cpu().numpy())

    # Clip predictions to the valid [0, 10] range
    predictions = np.clip(predictions, 0.0, 10.0)

    # Generate row_id ("test-000000", "test-000001", ...) if not already in test.csv
    if "row_id" in test_df.columns:
        row_ids = test_df["row_id"]
    else:
        row_ids = [f"test-{i:06d}" for i in range(len(test_df))]

    # Build and save final submission
    submission = pd.DataFrame({
        "row_id": row_ids,
        "result": predictions
    })
    submission.to_csv("sample_submission.csv", index=False)
    print(f"Saved submission.csv with {len(submission)} rows.")
    print(submission.head())

if __name__ == "__main__":
    create_submission()
