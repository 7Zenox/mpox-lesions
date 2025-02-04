import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard integration
import os

# Import your dataset and model
from dataloader import load_data, preprocess_data  # Replace with your dataset module

# from your_model_module import get_model  # Replace with your model module

# Define constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
CLASS_NAMES = ["Chickenpox", "Cowpox", "Healthy", "HFMD", "Measles", "Monkeypox"]


# Define training function
def train_one_epoch(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(
        tqdm(dataloader, desc="Training", leave=False)
    ):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Log training loss to TensorBoard for each batch
        writer.add_scalar(
            "Training Loss/Batch", loss.item(), epoch * len(dataloader) + batch_idx
        )

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total

    # Log epoch-level training metrics to TensorBoard
    writer.add_scalar("Training Loss/Epoch", epoch_loss, epoch)
    writer.add_scalar("Training Accuracy/Epoch", epoch_accuracy, epoch)

    return epoch_loss, epoch_accuracy


# Define validation function
def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct / total

    # Log validation metrics to TensorBoard
    writer.add_scalar("Validation Loss/Epoch", epoch_loss, epoch)
    writer.add_scalar("Validation Accuracy/Epoch", epoch_accuracy, epoch)

    return epoch_loss, epoch_accuracy


# Main training script
def train_model(
    model, train_loader, val_loader, num_epochs, learning_rate, save_path, log_dir
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train for one epoch
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, writer, epoch
        )
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        # Validate
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, DEVICE, writer, epoch
        )
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training complete.")
    writer.close()  # Close the TensorBoard writer
