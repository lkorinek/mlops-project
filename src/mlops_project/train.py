# Pytorch and Pytorch-lighning packages
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Logging
import hydra
from hydra.utils import to_absolute_path
import wandb
import logging
from omegaconf import OmegaConf

# import from project
from model import Model
from data import load_chest_xray_data


def set_seed(seed: int):
    """
    Sets the seed for reproducibility in PyTorch with GPUs and CPU
    Args:
        seed (int): specific value.
    """
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi GPUs
    torch.backends.cudnn.deterministic = True  # Make CUDA deterministic
    torch.backends.cudnn.benchmark = False  # Disable CUDA autotuning


log = logging.getLogger(__name__)


@hydra.main(config_path=to_absolute_path("configs/train_config"), config_name="default_config.yaml", version_base="1.1")
def train(config) -> None:
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment  # loading hyperparameters
    set_seed(hparams["seed"])  # setting reproducible seed
    # Wandb setup for project
    wandb.finish()
    wandb_logger = WandbLogger()
    wandb.init(
        project="X-Ray - Classification of Pneumonia",
        config={
            "lr": hparams["lr"],
            "weight_decay": hparams["wd"],
            "batch_size": hparams["batch_size"],
            "epochs": hparams["n_epochs"],
        },
    )

    # Pytorch-lightning initialize model from model.py
    model = Model(model_name=hparams["model_name"], num_classes=1, lr=hparams["lr"], wd=hparams["wd"])
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # For output logging
    print(
        f"lr = {hparams['lr']}, weight_decay = {hparams['wd']}, batch_size={hparams['batch_size']}, epochs={hparams['n_epochs']}"
    )
    wandb.config.update(
        {
            "lr": hparams["lr"],
            "weight_decay": hparams["wd"],
            "batch_size": hparams["batch_size"],
            "epochs": hparams["n_epochs"],
        }
    )

    # Using absolute path to ensure working dir is correct
    data_dir = os.path.join("data", "processed")
    trainset, testset, valset = load_chest_xray_data(to_absolute_path(data_dir))

    # Dataloader for training and testing set
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=hparams["batch_size"],
        shuffle=True,
        num_workers=hparams["num_workers"],
        persistent_workers=hparams["persistent_workers"],
    )
    test_dataloader = torch.utils.data.DataLoader(
        testset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        persistent_workers=hparams["persistent_workers"],
    )
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=hparams["batch_size"],
        shuffle=False,
        num_workers=hparams["num_workers"],
        persistent_workers=hparams["persistent_workers"],
    )

    # Saving the trained model in path models in the case of more models of the same model a suffix of "-vx" where x is version number will be added to trained model.
    model_save_path = to_absolute_path("models")
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename=f"trained_{hparams['model_name']}",
        save_top_k=1,  # Save only the best model
        verbose=True,
        monitor="val_loss",
        mode="min",  # Save when validation loss is minimized
        every_n_epochs=1,
    )
    # Trainer with WANDB logging
    trainer = pl.Trainer(
        max_epochs=hparams["n_epochs"],
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)
    wandb.finish()


if __name__ == "__main__":
    train()
