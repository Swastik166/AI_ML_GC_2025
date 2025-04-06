# data/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, List, Dict
import numpy as np 

class TrainValDataset(Dataset):
    """
    Dataset for training and validation folders where label is in filename.
    For training (is_train=True), creates `num_augmentations` augmented
    copies of each original image on the fly.
    """

    def __init__(self,
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 label_to_int: Optional[Dict[str, int]] = None,
                 is_train: bool = False, # Flag to indicate if this is for training
                 num_augmentations: int = 1 # Number of augmented copies per image for training
                ):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.num_augmentations = max(1, num_augmentations) if self.is_train else 1 

        self.original_image_files: List[str] = []
        self.original_labels: List[str] = [] 

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        print(f"Loading data from: {data_dir}")
        for filename in sorted(os.listdir(data_dir)):
            if filename.lower().endswith(('.jpg')):
                try:
                    label_part = filename.split('.')[0]
                    _ = int(label_part) 
                    self.original_labels.append(label_part)
                    self.original_image_files.append(os.path.join(data_dir, filename))
                except (IndexOutOfBoundsError, ValueError):
                    print(f"Warning: Could not parse label from filename '{filename}'. Skipping.")

        if not self.original_image_files:
             raise ValueError(f"No valid image files found in {data_dir}")

        print(f"Found {len(self.original_image_files)} original images.")
        if self.is_train and self.num_augmentations > 1:
             print(f"Will generate {self.num_augmentations} augmented versions per image during training.")

        # Create or use label_to_int mapping based on *original* labels
        if label_to_int is None:
            unique_labels = sorted(list(set(self.original_labels)))
            self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
            print(f"Created label map with {len(self.label_to_int)} classes.")
        else:
            self.label_to_int = label_to_int
            print(f"Using provided label map with {len(self.label_to_int)} classes.")


        self.original_int_labels = [self.label_to_int[lbl] for lbl in self.original_labels]

        # Store the inverse mapping
        self.int_to_label = {v: k for k, v in self.label_to_int.items()}

    def __len__(self) -> int:
        if self.is_train:
            return len(self.original_image_files) * self.num_augmentations
        else:
            return len(self.original_image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.is_train:
            original_idx = idx // self.num_augmentations
        else:
            original_idx = idx

        img_path = self.original_image_files[original_idx]
        label = self.original_int_labels[original_idx]

        try:

            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error opening image {img_path}: {e}. Returning dummy data.")
             print('=' * 100)
             print('=' * 100)
             


        if self.transform:
            image = self.transform(image)

        return image, label

    def get_num_classes(self) -> int:
        return len(self.label_to_int)

    def get_label_maps(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        return self.label_to_int, self.int_to_label


class TestDataset(Dataset):
    """Dataset for the test folder where filenames are like image1.jpg. (No changes needed here)"""

    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files: List[str] = []
        self.image_ids: List[str] = [] 

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        print(f"Loading test data from: {data_dir}")
        for filename in sorted(os.listdir(data_dir)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_ids.append(filename)
                self.image_files.append(os.path.join(data_dir, filename))

        if not self.image_files:
             raise ValueError(f"No valid image files found in {data_dir}")

        print(f"Found {len(self.image_files)} test images.")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_files[idx]
        image_id = self.image_ids[idx] 

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"Error opening image {img_path}: {e}. Returning dummy data.")
             print('=' * 100)
             print('=' * 100)

        if self.transform:
            image = self.transform(image)

        return image, image_id 