import torch
import pytest
import timm
from pytorch_lightning import Trainer

import sys
import os

sys.path.append(os.path.realpath("./src/mlops_project"))
from model import Simple_Network, Model

# Check if running in CI (GitHub Actions sets the `CI` environment variable to "true")
IN_GITHUB_ACTIONS = os.getenv("CI", "false").lower() == "true"

# Model test cases (name and number of classes)
model_test_cases = [
    ("simple", 1),
    ("densenet121", 1),
    ("resnet50", 1),
    ("vgg16", 1),
]

# Expected models for the corresponding test cases
expected_models = [
    Simple_Network,
    timm.models.DenseNet,
    timm.models.ResNet,
    timm.models.VGG,
]


@pytest.fixture
def mock_input():
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def mock_labels():
    return torch.randint(0, 2, (4,)).float()  # Binary labels


# Combine the test cases and expected models
@pytest.mark.parametrize(
    "model_name, num_classes, expected_model",
    [(*case, expected_model) for case, expected_model in zip(model_test_cases, expected_models)],
)
def test_model_initialization(model_name, num_classes, expected_model):
    """Test if the model initializes with our chosen models"""
    model = Model(model_name=model_name, num_classes=num_classes)
    assert model is not None, f"Model: {model_name} failed to get initialize."

    assert isinstance(model.model, expected_model), (
        f"Model type mismatch for {model_name}. Expected {expected_model}, got {type(model.model)}."
    )


def test_simple_network_forward(mock_input):
    """Test a forward pass through our simple model"""
    model = Simple_Network()
    mock_input = torch.randn(2, 3, 224, 224)
    output = model(mock_input)
    assert output.shape == (2, 1), f"Unexpected output shape {output.shape}, expected (2, 1)."


@pytest.mark.parametrize("model_name, num_classes", model_test_cases)
def test_model_forward(model_name, num_classes, mock_input):
    """Test a forward pass through the PyTorch-Lightning model."""
    model = Model(model_name=model_name, num_classes=num_classes)
    output = model(mock_input)
    assert output.shape == (4, 1), f"Unexpected output shape {output.shape}, expected (4, 1)."


@pytest.mark.parametrize("model_name, num_classes", model_test_cases)
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_training_step_with_trainer(model_name, num_classes, mock_input, mock_labels):
    """Test the training step using a PyTorch Lightning Trainer."""
    model = Model(model_name=model_name, num_classes=num_classes)
    trainer = Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        limit_train_batches=1,  # Only one training step
        limit_val_batches=1,  # Only one validation step
    )  # Simplified trainer for testing

    # Dummy trainset
    train_dataset = torch.utils.data.TensorDataset(mock_input, mock_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=3)

    # Dummy valset
    val_dataset = torch.utils.data.TensorDataset(mock_input, mock_labels)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, num_workers=3)

    # Pass val dataloader to the trainer
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    assert model.trainer is not None, "Trainer was not passed to the model during training."


@pytest.mark.parametrize("model_name, num_classes", model_test_cases)
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_validation_step_with_trainer(model_name, num_classes, mock_input, mock_labels):
    """Test the validation step using a PyTorch Lightning Trainer."""
    model = Model(model_name=model_name, num_classes=num_classes)
    trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False)  # Simplified trainer for testing

    dataset = torch.utils.data.TensorDataset(mock_input, mock_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4, persistent_workers=True)

    trainer.validate(model, dataloaders=dataloader)
    assert model.trainer is not None, "Trainer was not passed to the model during validation procedure"


@pytest.mark.parametrize("model_name, num_classes", model_test_cases)
def test_optimizer(model_name, num_classes):
    """Test optimizer setup."""
    model = Model(model_name=model_name, num_classes=num_classes, lr=1e-3, wd=1e-4)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer is not torch.optim.Adam."
