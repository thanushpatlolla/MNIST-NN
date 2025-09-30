import matplotlib.pyplot as plt

class Plot:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.ax2 = None
    def plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        if self.ax is None or self.ax2 is None:
            self.fig, (self.ax, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
            plt.ion()  # turn on interactive mode
            plt.show(block=False)

        self.ax.clear()
        self.ax.plot(train_losses, label="train loss")
        self.ax.plot(val_losses, label="val loss")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss")
        self.ax.legend()

        self.ax2.clear()
        self.ax2.plot(train_accuracies, label="train acc")
        self.ax2.plot(val_accuracies, label="val acc")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.set_title("Accuracy")
        self.ax2.set_ylim(0.8, 1.0)
        self.ax2.legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def final_plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.ioff()
        self.plot(train_losses, val_losses, train_accuracies, val_accuracies)
        plt.show()

