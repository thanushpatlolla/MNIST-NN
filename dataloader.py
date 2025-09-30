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
        
        data_path = f'./data/{self.dataset.upper()}'
        download_needed = not (os.path.exists(data_path) and os.listdir(data_path))
        
        if download_needed:
            print(f"Downloading {self.dataset.upper()} dataset...")
        else:
            print(f"Found existing {self.dataset.upper()} data, skipping download")
        
        # Get the correct dataset class
        dataset_class = getattr(torchvision.datasets, self.dataset.upper())
        
        train_set = dataset_class(
            './data', train=True, download=download_needed,
            transform=torchvision.transforms.Compose(train_transforms))
        test_set = dataset_class(
            './data', train=False, download=download_needed,
            transform=torchvision.transforms.Compose(test_transforms))

        x_train = []
        y_train = []
        for i in range(len(train_set)):
            data, label = train_set[i]
            x_train.append(data.numpy())
            y_train.append(label)

        x_test = []
        y_test = []
        for i in range(len(test_set)):
            data, label = test_set[i]
            x_test.append(data.numpy())
            y_test.append(label)

        x_train = np.array(x_train).reshape(len(x_train), -1)
        y_train = np.array(y_train)
        x_test = np.array(x_test).reshape(len(x_test), -1)
        y_test = np.array(y_test)

        return (x_train, y_train), (x_test, y_test)