import pytorch_lightning as pl
from model import Model_0, Model_1, Model_2
from dataset import TreeDataModule
import config
from lightning.pytorch.loggers import NeptuneLogger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateFinder,
    RichProgressBar,
)
import time


if __name__ == "__main__":
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZmM3ZjJhOS04MWM5LTQ3NWEtOGUzOS03YjczYTgyMWI5M2EifQ==",
        project="mpeaco/COMP3000-tree-detection",
    )
    model_0 = Model_0(
        input_shape=12288,
        hidden_units=10,
        output_shape=2,
        learning_rate=config.LEARNING_RATE,
    )
    model_1 = Model_1(
        input_shape=12288,
        hidden_units=10,
        output_shape=2,
        learning_rate=config.LEARNING_RATE,
    )
    model_2 = Model_2(
        input_shape=3,
        hidden_units=10,
        output_shape=2,
        learning_rate=config.LEARNING_RATE,
    )

    data_module = TreeDataModule(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = pl.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        logger=neptune_logger,
        callbacks=[
            RichProgressBar(),
            LearningRateFinder(1e-9, 1e-1),
        ],
    )

    start_time = time.time()

    trainer.fit(model_2, data_module)
    trainer.validate(model_2, data_module)

    end_time = time.time()
    time_taken = end_time - start_time
    print("\nElapsed time: {}".format(time_taken, "\n"))

# EarlyStopping(monitor="val_loss")
