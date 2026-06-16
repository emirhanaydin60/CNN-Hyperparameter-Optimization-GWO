from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils import seed_worker_factory


@dataclass
class DataBundle:
    train_dataset: object
    val_dataset: object
    test_dataset: object
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_channels: int
    img_size: int
    num_classes: int


def _build_dataset(dataset: str, data_dir: str):
    dataset_key = dataset.lower()

    if dataset_key in ("fashionmnist", "fashion-mnist"):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        in_channels = 1
        img_size = 28
        num_classes = 10
    elif dataset_key == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        in_channels = 1
        img_size = 28
        num_classes = 10
    elif dataset_key == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        in_channels = 3
        img_size = 32
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return train_dataset, test_dataset, in_channels, img_size, num_classes


def build_data_bundle(
    dataset: str,
    data_dir: str,
    batch_size: int,
    train_size: int,
    val_size: int,
    seed: int,
    num_workers: int,
    subset_size=None,
):
    train_dataset, test_dataset, in_channels, img_size, num_classes = _build_dataset(dataset, data_dir)

    if train_size + val_size > len(train_dataset):
        raise ValueError("train_size + val_size must be <= size of the training split")

    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=split_generator)

    if subset_size is not None and subset_size < len(train_subset):
        subset_generator = torch.Generator().manual_seed(seed + 1)
        train_subset, _ = random_split(train_subset, [subset_size, len(train_subset) - subset_size], generator=subset_generator)

    loader_generator = torch.Generator().manual_seed(seed)
    seed_worker = seed_worker_factory(seed)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DataBundle(
        train_dataset=train_subset,
        val_dataset=val_subset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        in_channels=in_channels,
        img_size=img_size,
        num_classes=num_classes,
    )
