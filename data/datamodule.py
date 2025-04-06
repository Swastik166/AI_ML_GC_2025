# data/datamodule.py
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict

from .dataset import TrainValDataset, TestDataset
from utils.augmentations import get_train_transforms, get_val_test_transforms

class GCDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: int = 224,
                 num_augmentations: int = 1,
                 strong_augmentations: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size


        self.train_dir = os.path.join(data_dir, 'train')
        self.val_dir = os.path.join(data_dir, 'val')
        self.test_dir = os.path.join(data_dir, 'test')


        if not os.path.isdir(self.train_dir):
             raise FileNotFoundError(f"Training directory not found: {self.train_dir}")
        if not os.path.isdir(self.val_dir):
             raise FileNotFoundError(f"Validation directory not found: {self.val_dir}")

        
        self.train_transforms = get_train_transforms(self.image_size, strong_augmentations)
        self.val_test_transforms = get_val_test_transforms(self.image_size)

        self.train_dataset: Optional[TrainValDataset] = None
        self.val_dataset: Optional[TrainValDataset] = None
        self.test_dataset: Optional[TestDataset] = None

        self.label_to_int: Optional[Dict[str, int]] = None
        self.int_to_label: Optional[Dict[int, str]] = None
        self.num_classes: Optional[int] = None

        
        self.save_hyperparameters()


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            # Create train dataset: pass is_train=True and num_augmentations
            self.train_dataset = TrainValDataset(
                self.train_dir,
                transform=self.train_transforms,
                is_train=True,
                num_augmentations=self.hparams.num_augmentations
            )
            self.label_to_int, self.int_to_label = self.train_dataset.get_label_maps()
            self.num_classes = self.train_dataset.get_num_classes()

            
            self.val_dataset = TrainValDataset(
                self.val_dir,
                transform=self.val_test_transforms,
                label_to_int=self.label_to_int,
                is_train=False 
                
            )
            if self.val_dataset.get_num_classes() != self.num_classes:
                 print("Warning: Number of classes in validation set files may differ from training set!")


        if stage == 'test' or stage == 'predict' or stage is None:
             if os.path.isdir(self.test_dir):
                
                self.test_dataset = TestDataset(self.test_dir, transform=self.val_test_transforms)
             else:
                 print(f"Test directory {self.test_dir} not found. Cannot create test/predict dataloader.")
                 self.test_dataset = None

        print(f"Setup complete for stage: {stage}")
        if self.num_classes:
            print(f"Number of classes detected: {self.num_classes}")
        if self.train_dataset and self.hparams.num_augmentations > 1:
            print(f"Effective training dataset size (with augmentations): {len(self.train_dataset)}")


    def train_dataloader(self):
        if not self.train_dataset:
             self.setup('fit')
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        if not self.val_dataset:
             self.setup('fit')
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        print("Warning: test_dataloader() called, but test set has no labels. Use predict_dataloader() for inference.")
        if not self.test_dataset:
             self.setup('test')
        if self.test_dataset:
             return DataLoader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0
            )
        else: return None

    def predict_dataloader(self):
        if not self.test_dataset:
             self.setup('predict')
        if self.test_dataset is None:
             print("Cannot create predict dataloader: Test dataset not available.")
             return None
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0
        )