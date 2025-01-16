import os
import torch
from torchvision import datasets, transforms
import typer
from typing import Tuple

def normalize(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images by subtracting the mean and dividing by the standard deviation.

    Args:
        images (torch.Tensor): Tensor of images to be normalized.

    Returns:
        torch.Tensor: Normalized images.
    """
    return (images - images.mean()) / images.std()

def preprocess_data(chest_xray: str, processed_dir: str) -> None:
    """
    Process raw chest x-ray data and save the processed data to a directory.

    Args:
        chest_xray (str): Path to the raw chest x-ray data.
        processed_dir (str): Path to the directory where processed data will be saved.
    """
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
        images, targets = [], []
        for img, label in dataset:
            images.append(img)
            targets.append(label)

        images_tensor = torch.stack(images)
        targets_tensor = torch.tensor(targets)

        images_tensor = normalize(images_tensor)

        torch.save(images_tensor, os.path.join(processed_dir, f"{prefix}_images.pt"))
        torch.save(targets_tensor, os.path.join(processed_dir, f"{prefix}_target.pt"))

    process_and_save(os.path.join(chest_xray, "train"), "train")
    process_and_save(os.path.join(chest_xray, "test"), "test")

def load_chest_xray_data(processed_dir: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load and return train and test datasets for chest x-ray data.

    Args:
        processed_dir (str): Path to the directory containing processed data.

    Returns:
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: Train and test datasets.
    """
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    print(f"Loaded {len(train_set)} training samples and {len(test_set)} test samples.")

    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess_data)

