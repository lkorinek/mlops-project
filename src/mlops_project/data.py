import os
import torch
from torchvision import datasets, transforms
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(chest_xray: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    # Create the processed_dir if it does not exist
    os.makedirs(processed_dir, exist_ok=True)

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ]
    )

    def process_and_save(data_path: str, prefix: str) -> None:
        """Helper function to process and save the data."""
        dataset = datasets.ImageFolder(data_path, transform=transform)
        images, targets = [], []
        for img, label in dataset:
            images.append(img)
            targets.append(label)

        images = torch.stack(images)
        targets = torch.tensor(targets)

        images = normalize(images)

        torch.save(images, os.path.join(processed_dir, f"{prefix}_images.pt"))
        torch.save(targets, os.path.join(processed_dir, f"{prefix}_target.pt"))

    process_and_save(os.path.join(chest_xray, "train"), "train")
    process_and_save(os.path.join(chest_xray, "test"), "test")


def load_chest_xray_data(processed_dir: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for chest x-ray."""
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    # Create datasets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
