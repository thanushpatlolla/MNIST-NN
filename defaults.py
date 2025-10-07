from dataloader import DataLoader
from nn import NN
from activations import ReLU, GeLU
from optimizer import AdamW


class defaults:
    def mnist():
        mnist_dataloader = DataLoader(augmentation=False)
        training_set, test_data = mnist_dataloader.get_data()

        MNIST_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]))
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
