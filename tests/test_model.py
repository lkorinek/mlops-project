import torch
import pytest
import timm
from pytorch_lightning import Trainer

import sys
import os

sys.path.append(os.path.realpath("./src/mlops_project"))
from model import Simple_Network, Model


# Test the functions using all the models in one run.
@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_model_initialization(model_name, num_classes):
    """Test if the model initializes with our chosen models"""
    model = Model(model_name=model_name, num_classes=num_classes)
    assert model is not None, f"Model: {model_name} failed to get initialize."

    valid_model_types = {
        "simple": Simple_Network,
        "resnet50": timm.models.ResNet,
        "densenet121": timm.models.DenseNet,
        "vgg16": timm.models.VGG,
    }
    assert model_name in valid_model_types, f"Unexpected model name: {model_name}"
    assert isinstance(model.model, valid_model_types[model_name]), (
        f"Model type mismatch for {model_name}. Expected {valid_model_types[model_name]}, got {type(model.model)}."
    )


@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_simple_network_forward(model_name, num_classes):
    """Test a forward pass through our simple model"""
    model = Simple_Network()
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1), f"Unexpected output shape {output.shape}, expected (2, 1)."


@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_model_forward(model_name, num_classes):
    """Test a forward pass through the PyTorch-Lightning model."""
    model = Model(model_name=model_name, num_classes=num_classes)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (4, 1), f"Unexpected output shape {output.shape}, expected (4, 1)."


@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_training_step_with_trainer(model_name, num_classes):
    """Test the training step using a PyTorch Lightning Trainer."""
    model = Model(model_name=model_name, num_classes=num_classes)
    trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False)  # Simplified trainer for testing

    # Dummy trainset
    dummy_data = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    dummy_labels = torch.randint(0, 2, (4,)).float()  # Binary labels
    train_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, num_workers=4, persistent_workers=True)

    # Dummy valset
    val_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, num_workers=4, persistent_workers=True)

    # Pass val dataloader to the trainer
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    assert model.trainer is not None, "Trainer was not passed to the model during training."


@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_validation_step_with_trainer(model_name, num_classes):
    """Test the validation step using a PyTorch Lightning Trainer."""
    model = Model(model_name=model_name, num_classes=num_classes)
    trainer = Trainer(max_epochs=1, enable_checkpointing=False, logger=False)  # Simplified trainer for testing

    dummy_data = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    dummy_labels = torch.randint(0, 2, (4,)).float()  # Binary labels
    dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=4, persistent_workers=True)

    trainer.validate(model, dataloaders=dataloader)
    assert model.trainer is not None, "Trainer was not passed to the model during validation procedure"


@pytest.mark.parametrize(
    "model_name, num_classes",
    [
        ("simple", 1),
        ("densenet121", 1),
        ("resnet50", 1),
        ("vgg16", 1),
    ],
)
def test_optimizer(model_name, num_classes):
    """Test optimizer setup."""
    model = Model(model_name=model_name, num_classes=num_classes, lr=1e-3, wd=1e-4)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer is not torch.optim.Adam."
