import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Define paths to dataset folders
data_path = "dataset"  # Replace with your dataset path
original_images_path = os.path.join(
    data_path, "Original Images", "Original Images", "FOLDS"
)
augmented_images_path = os.path.join(
    data_path, "Augmented Images", "Augmented Images", "FOLDS_AUG"
)

# Define image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# Class names
CLASS_NAMES = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]


# Helper function to load image paths and labels
def load_images_from_directory(base_path, fold, dataset_type, augmented=False):
    images = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        folder_path = os.path.join(
            base_path,
            (
                f"fold{fold}_AUG/train/{class_name}"
                if augmented
                else f"fold{fold}/{dataset_type}/{class_name}"
            ),
        )
        if not os.path.exists(folder_path):
            print(f"Warning: Path does not exist: {folder_path}")
            continue

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if os.path.isfile(img_path):
                images.append((img_path, class_index))
    return images


# Custom PyTorch Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3 channels
        if self.transform:
            image = self.transform(image)
        return image, label


# Load original and augmented data
def load_data():
    datasets = {}
    for fold in range(1, 6):
        print(f"Loading data for fold {fold}...")

        # Load original data
        datasets[f"fold{fold}_train"] = load_images_from_directory(
            original_images_path, fold, "train"
        )
        datasets[f"fold{fold}_val"] = load_images_from_directory(
            original_images_path, fold, "valid"
        )
        datasets[f"fold{fold}_test"] = load_images_from_directory(
            original_images_path, fold, "test"
        )

        # Load augmented data
        datasets[f"fold{fold}_train_aug"] = load_images_from_directory(
            augmented_images_path, fold, "train", augmented=True
        )

    return datasets


# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),  # Convert image to tensor and normalize to [0, 1]
    ]
)


# Preprocess and create PyTorch DataLoader
def preprocess_data(data, batch_size=BATCH_SIZE):
    if not data:
        raise ValueError(
            "Dataset is empty. Ensure the data paths and structure are correct."
        )

    dataset = CustomImageDataset(data, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    return dataloader


# Load datasets
datasets = load_data()

# Example: Preparing fold 1 train and validation datasets
try:
    fold1_train_loader = preprocess_data(datasets["fold1_train"])
    fold1_val_loader = preprocess_data(datasets["fold1_val"])
    print("Data loading and preprocessing complete.")
except ValueError as e:
    print(e)
