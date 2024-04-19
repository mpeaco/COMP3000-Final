import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import datasets


class TreeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = num_workers
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        full_dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=transform,
        )

        # Split the data into training and test data sets
        self.test_data = None
        self.train_data = None
        test_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - test_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, test_size]
        )

    def train_dataloader(self):
        # Create dataloader using the train data and return
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        # Create a dataloader for the validation set
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=False,
        )
