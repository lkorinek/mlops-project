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
    # Define the transformations (resize and normalization)
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Resize to a consistent shape
        transforms.ToTensor(),  # Convert to tensor
    ])
    
    # Load the datasets using ImageFolder (this will automatically handle the directories)
    train_dataset = datasets.ImageFolder(os.path.join(chest_xray, 'train'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(chest_xray, 'test'), transform=transform)

    # Initialize lists to store the images and labels
    train_images, train_target = [], []
    for img, label in train_dataset:
        train_images.append(img)
        train_target.append(label)

    # Convert lists to tensors
    train_images = torch.stack(train_images)  # Now, all images will be resized and stacked
    train_target = torch.tensor(train_target)

    test_images, test_target = [], []
    for img, label in test_dataset:
        test_images.append(img)
        test_target.append(label)

    test_images = torch.stack(test_images)  # Now, all images will be resized and stacked
    test_target = torch.tensor(test_target)

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save the processed data
    torch.save(train_images, os.path.join(processed_dir, 'train_images.pt'))
    torch.save(train_target, os.path.join(processed_dir, 'train_target.pt'))
    torch.save(test_images, os.path.join(processed_dir, 'test_images.pt'))
    torch.save(test_target, os.path.join(processed_dir, 'test_target.pt'))

def load_chest_xray_data(processed_dir: str) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for chest x-ray."""
    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"), weights_only=True)
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"), weights_only=True)
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"), weights_only=True)
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"), weights_only=True)

    # Create datasets
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    
    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess_data)
