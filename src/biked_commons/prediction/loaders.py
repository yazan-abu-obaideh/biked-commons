import pandas as pd
import numpy as np
import os
import requests
from tqdm import tqdm

def load_validity():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/validity_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/validity_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/validity_Y_train.csv', index_col=0)
    return X_train, Y_train, X_test
def load_structure():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/structure_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/structure_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/structure_Y_train.csv', index_col=0)
    return X_train, Y_train, X_test

def load_aero():
    X_train = pd.read_csv('../../resources/datasets/split_datasets/aero_X_train.csv', index_col=0)
    X_test = pd.read_csv('../../resources/datasets/split_datasets/aero_X_test.csv', index_col=0)
    Y_train = pd.read_csv('../../resources/datasets/split_datasets/aero_Y_train.csv', index_col=0)
    return X_train, Y_train, X_test

def load_clip():
    check_download_CLIP_data()
    X_train = pd.read_csv("../../resources/datasets/split_datasets/CLIP_X_train.csv", index_col=0)
    Y_train = np.load("../../resources/datasets/split_datasets/CLIP_Y_train.npy")
    X_test = pd.read_csv("../../resources/datasets/split_datasets/CLIP_X_test.csv", index_col=0)
    return X_train, Y_train, X_test

def download_file(file_url, file_path):
    """Downloads a file with a progress bar if it doesn't exist locally."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists

    response = requests.get(file_url, stream=True)
    
    if response.status_code == 200:
        file_size = int(response.headers.get('content-length', 0))  # Get file size if available
        chunk_size = 1024  # 1 KB per chunk

        with open(file_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(file_path)}",
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"✅ Download complete: {file_path}")
    else:
        print(f"❌ Error downloading {file_path}: {response.status_code}")

def check_download_CLIP_data():
    # File paths and URLs
    x_id = "10541435"
    x_url = f"https://dataverse.harvard.edu/api/access/datafile/{x_id}"
    x_file = "../../resources/datasets/split_datasets/CLIP_X_train.csv"

    y_id = "10992683"
    y_url = f"https://dataverse.harvard.edu/api/access/datafile/{y_id}"
    y_file = "../../resources/datasets/split_datasets/CLIP_Y_train.npy"

    # Check and download X
    if not os.path.exists(x_file):
        print(f"⚠️  {os.path.basename(x_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(x_url, x_file)
    else:
        print(f"✅ {os.path.basename(x_file)} already exists in datasets folder. Skipping download.")

    # Check and download Y
    if not os.path.exists(y_file):
        print(f"⚠️  {os.path.basename(y_file)} not found in datasets folder. Performing first-time download from Harvard Dataverse...")
        download_file(y_url, y_file)
    else:
        print(f"✅ {os.path.basename(y_file)} already exists in datasets folder. Skipping download.")