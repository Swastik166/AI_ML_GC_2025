# utils/augmentations.py
import torch
from torchvision.transforms import v2 as transforms 


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int, strong_augmentations: bool = False):
    """
    Creates a composition of randomized training transforms.
    Each call to this transform on the same input image will likely
    produce a different output due to the random operations.

    Args:
        image_size (int): Target image size (height and width).

    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    transform_list = []

    # --- Strong augmentations (optional) ---
    if strong_augmentations:
        transform_list.extend([
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3))], p=0.5),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomGrayscale(p=0.1),
        ])

    # --- Basic augmentations ---
    transform_list.extend([
        transforms.RandomResizedCrop(size=image_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.07),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5), #check
        transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), value=IMAGENET_MEAN, p=0.2),

        # --- Final mandatory transforms ---
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print(f"Using Training Transforms: {[type(t).__name__ for t in transform_list]}")
    return transforms.Compose(transform_list)


def get_val_test_transforms(image_size: int):
    """
    Creates a composition of validation/testing transforms.
    No augmentation here, just resizing and normalization.

    Args:
        image_size (int): Target image size (height and width).

    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])