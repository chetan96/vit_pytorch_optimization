import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

def get_cifar_dataloaders(data_root='../data', batch_size=32, val_ratio=0.1):
    """
    Args:
        data_root: Path to directory where CIFAR-10 will be downloaded/stored
        batch_size: Batch size for dataloaders
        val_ratio: Fraction of training data for validation
    """
    # Create directory if it doesn't exist
    os.makedirs(data_root, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) #Cifar-10 standar normalization values
    ])
    
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    val_size = int(val_ratio * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader