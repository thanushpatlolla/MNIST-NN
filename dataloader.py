import torchvision
import numpy as np

class DataLoader():
    def __init__(self, dataset="mnist", augmentation=False):
        self.dataset = dataset
        self.augmentation = augmentation

    def get_data(self):
        # Base transforms (no augmentation)
        base_train_transforms = [torchvision.transforms.ToTensor()]
        test_transforms = [torchvision.transforms.ToTensor()]
        
        # Add normalization
        if self.dataset == "mnist":
            base_train_transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
            test_transforms.append(torchvision.transforms.Normalize((0.1307,), (0.3081,)))
        elif self.dataset == "cifar10":
            base_train_transforms.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
            test_transforms.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]))
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
        
        # Load original dataset without augmentation
        if self.dataset == "mnist":
            original_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(base_train_transforms))
            test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose(test_transforms))
        elif self.dataset == "cifar10":
            original_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(base_train_transforms))
            test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.Compose(test_transforms))
        
        # Get original training data
        x_train = np.stack([original_train_set[i][0].numpy().reshape(-1) for i in range(len(original_train_set))])
        y_train = np.array([original_train_set[i][1] for i in range(len(original_train_set))])
        
        # Add augmented data if augmentation is enabled
        if self.augmentation:
            # Create augmented transforms
            augmented_transforms = base_train_transforms.copy()
            if self.dataset == "mnist":
                augmented_transforms.insert(-2, torchvision.transforms.RandomCrop(28, padding=4, padding_mode="reflect"))
            elif self.dataset == "cifar10":
                augmented_transforms.insert(-2, torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'))
                augmented_transforms.insert(-2, torchvision.transforms.RandomHorizontalFlip(p=0.5))
            
            # Load augmented dataset
            if self.dataset == "mnist":
                augmented_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(augmented_transforms))
            elif self.dataset == "cifar10":
                augmented_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.Compose(augmented_transforms))
            
            # Get augmented training data
            x_augmented = np.stack([augmented_train_set[i][0].numpy().reshape(-1) for i in range(len(augmented_train_set))])
            y_augmented = np.array([augmented_train_set[i][1] for i in range(len(augmented_train_set))])
            
            # Concatenate original and augmented data
            x_train = np.concatenate([x_train, x_augmented], axis=0)
            y_train = np.concatenate([y_train, y_augmented], axis=0)
        
        # Get test data
        x_test = np.stack([test_set[i][0].numpy().reshape(-1) for i in range(len(test_set))])
        y_test = np.array([test_set[i][1] for i in range(len(test_set))])

        return (x_train, y_train), (x_test, y_test)