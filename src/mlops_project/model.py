import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
import torchmetrics
import typer

# Profiling
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler


class Simple_Network(nn.Module):
    def __init__(self):
        super(Simple_Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.flattened_size = 64 * 28 * 28  # here we have to pass 3 x 150 x 150 (RGB and transformed images)

        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Model(pl.LightningModule):
    def __init__(self, model_name: str, num_classes: int, lr: float = 1e-3, wd: float = 1e-4):
        super(Model, self).__init__()
        self.lr = lr  # Learning rate
        self.wd = wd  # Weight-decay
        self.num_classes = num_classes
        self.model = self.load_model(model_name, num_classes)
        self.criterion = nn.BCEWithLogitsLoss()  # With Logits binary cross entropy since we have 2 classes
        self.train_losses = []  # For computing train loss for each epoch (might be redundant)

        # Torch metrics for metric computation
        self.accuracy = torchmetrics.classification.Accuracy(task="binary", num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="binary", num_classes=num_classes)
        self.precision = torchmetrics.classification.Precision(task="binary", num_classes=num_classes)
        self.recall = torchmetrics.classification.Recall(task="binary", num_classes=num_classes)

    def load_model(self, model_name: str, num_classes: int):
        if model_name == "simple":
            model = Simple_Network()
        else:
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze the final layer
            if hasattr(model, "fc"):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, "classifier"):
                for param in model.classifier.parameters():
                    param.requires_grad = True
            elif hasattr(model, "head"):
                for param in model.head.parameters():
                    param.requires_grad = True

        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs.squeeze(), targets.float())
        self.train_losses.append(loss.item())
        self.accuracy(outputs.squeeze(), targets.float())
        self.log("train_acc_epoch", self.accuracy, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = sum(self.train_losses) / len(self.train_losses)
        self.log("avg_train_loss", avg_loss)  # take the average loss of the batches for each epoch
        self.train_losses.clear()

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs.squeeze(), targets.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        targets = targets.int()

        self.accuracy(preds.squeeze(), targets.float())
        self.precision(preds.squeeze().float(), targets.float())
        self.recall(preds.squeeze().float(), targets.float())
        self.f1(preds.squeeze().float(), targets.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_precision", self.precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", self.recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "val_loss": loss,
            "val_acc": self.accuracy,
            "val_precision": self.precision,
            "val_recall": self.recall,
            "val_f1": self.f1,
        }

    def test_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self(data)
        loss = self.criterion(outputs.squeeze(), targets.float())
        preds = (torch.sigmoid(outputs) > 0.5).int()
        targets = targets.int()

        self.accuracy(preds.squeeze(), targets.float())
        self.precision(targets, preds.squeeze())
        self.recall(targets, preds.squeeze())
        self.f1(targets, preds.squeeze())

        self.log("test_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_precision", self.precision, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_recall", self.recall, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("test_f1", self.f1, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return {
            "test_loss": loss,
            "test_acc": self.accuracy,
            "test_precision": self.precision,
            "test_recall": self.recall,
            "test_f1": self.f1,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)


def main(model_name: str = "simple", num_classes: int = 1, lr: float = 1e-3, wd: float = 1e-4):
    model = Model(model_name=model_name, num_classes=num_classes, lr=lr, wd=wd)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    typer.run(main)

    # Debugging stuff
    # from data import load_chest_xray_data
    # trainset, testset = load_chest_xray_data(r"data\processed") # Local path to processed data

    # # Dataloader for training and testing set
    # train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    # test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, persistent_workers=True)

    # # Python-lightning trainer that handles the training elements for us.
    # trainer = pl.Trainer(max_epochs=5, devices=1 if torch.cuda.is_available() else 0, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
    # trainer.fit(model, train_dataloader, trainer)
    # trainer.test(model, test_dataloader)

    # Profiling
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     model(dummy_input)
