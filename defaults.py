from dataloader import DataLoader
from nn import NN
from activations import ReLU, GeLU
from optimizer import AdamW


class defaults:
    def mnist():
        mnist_dataloader = DataLoader(augmentation=False)
        training_set, test_data = mnist_dataloader.get_data()

        MNIST_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]), 
                        epochs=25, hidden_dims=[256, 128], output_dim=10, act=ReLU(), lr_max=1e-3, 
                        lr_min=1e-5, warmup=2, batch_size=128, optimizer=AdamW, weight_decay=7.5e-3, dropout=0.3)
        MNIST_solver.train()

        return MNIST_solver
    def cifar10():
        cifar10_dataloader = DataLoader(augmentation=True)
        training_set, test_data = cifar10_dataloader.get_data()
        
        CIFAR10_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]), 
                  epochs=120, hidden_dims=[2048, 1024], act=GeLU(), lr_max=1e-3, warmup=6, 
                  weight_decay=0.03, dropout=0.5)
        CIFAR10_solver.train()
        return CIFAR10_solver
    def fashion_mnist():
        fashion_dataloader = DataLoader(dataset="fashion_mnist", augmentation=True)
        training_set, test_data = fashion_dataloader.get_data()

        FashionMNIST_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]),
                  epochs=30, hidden_dims=[256, 128], act=GeLU(), lr_max=1.5e-3, warmup=4,
                  weight_decay=0.02, dropout=0.4)
        FashionMNIST_solver.train()
        return FashionMNIST_solver
