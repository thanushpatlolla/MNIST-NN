# Neural Network from Scratch

A complete implementation of a neural network from scratch using only NumPy for image classification on MNIST, FashionMNIST, CIFAR-10 datasets. This project demonstrates the fundamentals of deep learning by implementing all core components manually. 

Taught me a lot about all the math behind core features used in most deep neural networks. Also helped me understand hyperparameter optimization. 

Does quite well on MNIST/FashionMNIST but struggles on CIFAR-10 as expected when we don't use a CNN.

MNIST Accuracy is around 98% 
FashionMNIST Accuracy is around 92%
CIFAR-10 Accuracy is around 59%

## Features

- **Complete Neural Network Implementation**: Forward pass, backpropagation, and training loop
- **Multiple Activation Functions**: ReLU and GeLU
- **Optimizers**: SGD and AdamW with gradient clipping
- **Regularization**: L2 regularization and Dropout
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Data Augmentation**: Random cropping and horizontal flipping
- **Input Normalization**: Dataset-specific normalization
- **Real-time Visualization**: Live plotting of training progress
- **Modular Design**: Clean separation of concerns across multiple files

## 📊 Performance

### MNIST Results
- **Settings**: 256 and 128 neuron hidden layers
- **Final Test Accuracy**: 97.94%
- **Training Set**: 60,000 samples
- **Validation Set**: 5,000 samples  
- **Test Set**: 5,000 samples

### CIFAR-10 Results
- Currently configured for CIFAR-10 with data augmentation
- Architecture: 1024 → 512 → 10 neurons
- Batch size: 128, Learning rate: 1e-3 to 1e-5 (cosine annealing)
- Weight decay: 0.03, Dropout: 0.5
- Epochs: 120 (warmup 5)

## 🏗️ Architecture

The neural network supports flexible architectures with configurable:
- Number of hidden layers
- Hidden layer dimensions
- Activation functions
- Output dimensions
- Dropout rates
- Regularization strength

## 📁 Project Structure

```
├── main.py              # Main training script
├── nn.py                # Neural network implementation
├── activations.py       # Activation functions (ReLU, GeLU)
├── loss.py              # Loss functions (Cross-entropy)
├── optimizer.py         # Optimizers (SGD, AdamW)
├── dataloader.py        # Data loading and preprocessing
├── plot.py              # Real-time plotting utilities
├── data/                # Dataset storage
│   ├── MNIST/          # MNIST dataset files
│   └── cifar-10-batches-py/  # CIFAR-10 dataset files
└── README.md           # This file
```


## 🔧 Configuration Options

### Neural Network Parameters
- `epochs`: Number of training epochs
- `hidden_dims`: List of hidden layer dimensions
- `lr_max`: Maximum learning rate (initial learning rate)
- `lr_min`: Minimum learning rate (for cosine annealing)
- `batch_size`: Mini-batch size
- `dropout`: Dropout probability (0.0 to 1.0)
- `weight_decay/L2 Regularization`: Weight decay strength for AdamW or L2 Regularization for standard SGD
- `optimizer`: Optimizer class (SGD or AdamW)
- `act`: Activation function (ReLU or GeLU)

### Data Loading Options
- `dataset`: "mnist" or "cifar10"
- `augmentation`: Enable/disable data augmentation

## 🧠 Key Implementation Details

### Forward Pass
- He initialization for hidden layers (variance = 2/fan_in)
- Glorot initialization for output layer (variance = 2/(fan_in + fan_out))
- ReLU/GeLU activation functions
- Dropout during training only (scaled by 1/(1-dropout))
- Softmax for final classification

### Backward Pass
- Manual backpropagation derivation
- Proper dropout gradient handling (scaled by 1/(1-dropout))
- Weight decay gradients (L2 regularization)
- Cross-entropy loss with softmax

### Training Features
- Mini-batch gradient descent
- Cosine annealing learning rate schedule with warmup
- Real-time loss and accuracy plotting
- Train/validation/test split (50/50 split of test set)
- AdamW optimizer with weight decay

## 📈 Learning Rate Schedule

The implementation uses a cosine annealing schedule with warmup:
- **Warmup**: Linear increase for first 2 epochs worth of steps
- **Annealing**: Cosine decay for remaining epochs
- **Minimum LR**: Configurable minimum learning rate (default: 1e-5)
- **Maximum LR**: Configurable maximum learning rate (default: 1e-3)

## 🎯 Regularization Techniques

1. **Weight Decay**: L2 regularization applied through AdamW optimizer
2. **Dropout**: Random neuron deactivation during training (scaled by 1/(1-dropout))
3. **Data Augmentation**: Random crops and horizontal flips (CIFAR-10 only)
4. **Input Normalization**: Dataset-specific mean/std normalization
5. **Learning Rate Scheduling**: Cosine annealing with warmup to prevent overfitting

## 🔍 Monitoring and Visualization

The training process includes:
- Real-time loss and accuracy plots
- Training vs validation metrics
- Final test accuracy reporting
- Interactive matplotlib visualization

## 🚧 Future Improvements

- [ ] Batch normalization implementation
- [ ] Early stopping mechanism
- [ ] Model checkpointing
- [ ] Hyperparameter tuning utilities
- [ ] CNN implementation for better CIFAR-10 performance
- [ ] Additional activation functions
- [ ] Learning rate finder
- [ ] Add options to output_layer
- [ ] Make each layer its own class (makes it easier to add batchnorm/layernorm/dropout/convolutions). Probably not going to do this and just do it in PyTorch.

## 📄 License

This project is open source and available under the MIT License.