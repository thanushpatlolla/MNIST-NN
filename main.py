from dataloader import DataLoader
from nn import NN

mnist_dataloader = DataLoader(augmentation=False)
training_set, test_data = mnist_dataloader.get_data()

MNIST_solver=NN(training_set, test_data, len(training_set[0][0]))
MNIST_solver.train(50)