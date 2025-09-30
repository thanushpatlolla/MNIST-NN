import torchvision
import numpy as np
import os

class DataLoader():
    def __init__(self, dataset="mnist", augmentation=False):
        self.dataset = dataset
        self.augmentation = augmentation

    def get_data(self):
        test_transforms=[]
        train_transforms=[]
        if self.augmentation:
            if self.dataset == "mnist":
                train_transforms.extend([
                    torchvision.transforms.RandomCrop(28, padding=4, padding_mode="reflect")
                ])
            elif self.dataset == "cifar10":
                train_transforms.extend([
                    torchvision.transforms.RandomCrop(32, padding=3, padding_mode='reflect'),
                    torchvision.transforms.RandomHorizontalFlip(p=0.5)
                ])
        train_transforms.append(torchvision.transforms.ToTensor())
        test_transforms.append(torchvision.transforms.ToTensor())
        if self.dataset == "mnist":
            #mean and std for mnist, searched it up
            train_transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
            test_transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
        elif self.dataset == "cifar10":
            #likewise for cifar10
            train_transforms.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
            test_transforms.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        
        if self.dataset == "mnist":
            train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(train_transforms))
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose(test_transforms))
        elif self.dataset == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(train_transforms))
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.Compose(test_transforms))
            
        x_train = np.stack([train_set[i][0].numpy().reshape(-1) for i in range(len(train_set))])
        y_train = np.array([train_set[i][1] for i in range(len(train_set))])
        x_test  = np.stack([test_set[i][0].numpy().reshape(-1) for i in range(len(test_set))])
        y_test  = np.array([test_set[i][1] for i in range(len(test_set))])

        return (x_train, y_train), (x_test, y_test)