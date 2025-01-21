import os
from typing import Tuple

import numpy as np
import torch
import typer
from torch.utils.data import Subset
from torchvision import datasets, transforms


def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Tensor of images to be normalized.

    Returns:
        torch.Tensor: Normalized images.
    """
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str, percentage: float = 1.0) -> None:
    """
    Process raw chest x-ray data and save the processed data to a directory.

    Args:
        raw_dir (str): Path to the raw chest x-ray data.
        processed_dir (str): Path to the directory where processed data will be saved.
        percentage (float): Percentage of images to be processed.
    """
    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be between 0 and 1.")

    os.makedirs(processed_dir, exist_ok=True)

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    def process_and_save(data_path: str, prefix: str) -> None:
        """
        Helper function to process and save images and targets.

        Args:
            data_path (str): Path to the data.
            prefix (str): Prefix for the saved files.
        """
        dataset = datasets.ImageFolder(data_path, transform=transform)

        # Select a subset of the dataset based on the specified percentage
        np.random.seed(42)
        total_samples = len(dataset)
        num_samples = int(total_samples * percentage)
        indices = np.random.choice(total_samples, num_samples, replace=False)
        subset = Subset(dataset, indices)

        images, targets = [], []
        for img, label in subset:
            images.append(img)
            targets.append(label)

        images_tensor = torch.stack(images)
        targets_tensor = torch.tensor(targets)

        images_tensor = normalize(images_tensor)

        torch.save(images_tensor, os.path.join(processed_dir, f"{prefix}_images.pt"))
        torch.save(targets_tensor, os.path.join(processed_dir, f"{prefix}_target.pt"))

    process_and_save(os.path.join(raw_dir, "train"), "train")
    process_and_save(os.path.join(raw_dir, "test"), "test")
    process_and_save(os.path.join(raw_dir, "val"), "val")


def load_chest_xray_data(processed_dir: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load and return train and test datasets for chest x-ray data.

    Args:
        processed_dir (str): Path to the directory containing processed data.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Train and test datasets.
    """
    # weights_only to combat user warning
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"), weights_only=True)
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"), weights_only=True)
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"), weights_only=True)
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"), weights_only=True)
    val_images = torch.load(os.path.join(processed_dir, "val_images.pt"), weights_only=True)
    val_target = torch.load(os.path.join(processed_dir, "val_target.pt"), weights_only=True)

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    val_set = torch.utils.data.TensorDataset(val_images, val_target)
    print(f"Loaded {len(train_set)} training samples, {len(test_set)} test samples and {len(val_set)} val samples")

    return train_set, test_set, val_set


if __name__ == "__main__":
    typer.run(preprocess_data)
