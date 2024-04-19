# This contains all models used throughout experiments
import pytorch_lightning as pl
from torch import nn, optim
import torch
import torchmetrics


class Model_0(pl.LightningModule):
    """This is a very basic model used as a bassline
    to demonstrate tree detection from sattelite imagery

    Args:
        pl (LightningModule): Inherits from pl.LightningModule
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate,
    ):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=input_shape, out_features=hidden_units
            ),  # number of features in a data sample (784 pixels)
            nn.Linear(in_features=hidden_units, out_features=output_shape),
        )
        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.layer_stack(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        return {"loss": loss, "logits": logits, "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Model_1(pl.LightningModule):
    """Second model to see if there is an improvement on the
    basic linear model using nn.Relu and dropout

    Args:
        pl (LightningModule): Inherits from pl.LightningModule
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate,
    ):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=input_shape, out_features=hidden_units
            ),  # number of features in a data sample (784 pixels)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU(),
        )
        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.layer_stack(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        return {"loss": loss, "logits": logits, "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


class Model_2(pl.LightningModule):
    """Implementaion of a basic CNN (TinyVGG)
    - 2 blocks
    - classifier

    Args:
        pl (LightningModule): Inherits from pl.LightningModule
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate,
    ):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=2560, out_features=2),
        )

        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        return {"loss": loss, "logits": logits, "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)


class Model_3(pl.LightningModule):
    """

    Args:
        pl (LightningModule): Inherits from pl.LightningModule
    """

    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        learning_rate,
    ):
        super().__init__()

        # Add code here for a better model -

        self.lr = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        accuracy = self.accuracy(logits, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
            }
        )
        return {"loss": loss, "logits": logits, "y": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr)
