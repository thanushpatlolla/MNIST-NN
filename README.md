Implemented a neural network from scratch in NumPy for image classification on the MNIST dataset. 
Pretty standard, used ReLU, He initialization, 2 hidden layers (256 and 128), standard mini-batch gradient descent. 
Overall quite informative on learning the intuition for backpropagation, but PyTorch is much more convenient
Final Test Accuracy: 97.94% (60000 in training set, 5000 in validation, 5000 in test)

Todo: 
Modularize
Try on other image datasets like CIFAR-10 (will show need for CNN)
Probably not needed for performance, but implement:
    Adam/AdamW optimizer, LR schedule
    Batchnorm (probably not deep enough to matter, but still)
    Dropout
    Normalize inputs
    Data augmentation (crop, flipping is bad for digits but useful for other image datasets)
    Early stopping