from dataloader import DataLoader
from nn import NN
from activations import ReLU, GeLU
from optimizer import AdamW

# mnist_dataloader = DataLoader(augmentation=False)
# training_set, test_data = mnist_dataloader.get_data()

# MNIST_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]))
# MNIST_solver.train()

cifar10_dataloader = DataLoader(dataset="cifar10", augmentation=True)
training_set, test_data = cifar10_dataloader.get_data()

CIFAR10_solver=NN(training_set=training_set, test_data=test_data, input_dim=len(training_set[0][0]), 
                  epochs=40, hidden_dims=[1024, 512], act=GeLU(), lr_max=1e-3, warmup=12, 
                  weight_decay=0.02, dropout=0.4)
CIFAR10_solver.train()