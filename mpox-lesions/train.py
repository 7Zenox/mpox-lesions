import argparse
import torch
from data_loader import load_data, preprocess_data
from model_loader import get_model
from train_utils import (
    train_model,
)  # Assuming train_model is defined as in the previous script

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a model on the dataset.")
parser.add_argument(
    "--model", type=str, required=True, help="Model to train: 'swin' or 'vnet'"
)
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--save_path",
    type=str,
    default="best_model.pth",
    help="Path to save the trained model",
)
parser.add_argument(
    "--log_dir", type=str, default="runs/experiment", help="TensorBoard log directory"
)
args = parser.parse_args()

# Load data
datasets = load_data()
train_loader = preprocess_data(datasets["fold1_train"], batch_size=args.batch_size)
val_loader = preprocess_data(datasets["fold1_val"], batch_size=args.batch_size)

# Initialize model
if args.model.lower() == "swin":
    model = get_model(model_name="swin", num_classes=6)
elif args.model.lower() == "vnet":
    model = get_model(model_name="vnet", input_channels=1, num_classes=6)
else:
    raise ValueError("Invalid model name. Choose 'swin' or 'vnet'.")

# Train the model
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=args.epochs,
    learning_rate=args.learning_rate,
    save_path=args.save_path,
    log_dir=args.log_dir,
)
