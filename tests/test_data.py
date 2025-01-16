import os
import torch
import pytest
import sys

sys.path.append(os.path.realpath("./src/mlops_project"))

from data import load_chest_xray_data

# Path to the processed data directory
x_ray_processed_dir = "data/processed"


@pytest.mark.skipif(not os.path.exists(x_ray_processed_dir), reason="Data files not found")
def test_data():
    # Pass the processed directory path to load_chest_xray_data
    train, test, val = load_chest_xray_data(x_ray_processed_dir)

    # Check each dataset (train, test)
    for dataset in [train, test, val]:
        for x, y in dataset:
            # Verify that the input images have the expected shape
            assert x.shape == (3, 224, 224), f"Unexpected shape {x.shape}, expected (3, 150, 150)"

            # Verify that the labels are within the expected range (0 or 1)
            assert y in [0, 1], f"Unexpected label {y}, expected 0 or 1"

    # Check if all target labels in set are unique and in [0, 1]
    train_targets = torch.unique(train.tensors[1])
    test_targets = torch.unique(test.tensors[1])
    val_targets = torch.unique(val.tensors[1])

    assert (train_targets == torch.tensor([0, 1])).all(), "Train targets are not exclusively 0 and 1"
    assert (val_targets == torch.tensor([0, 1])).all(), "Validation targets are not exclusively 0 and 1"
    assert (test_targets == torch.tensor([0, 1])).all(), "Test targets are not exclusively 0 and 1"
