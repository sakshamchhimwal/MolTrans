import matplotlib.pyplot as plt
import json

def make_loss_plots():
    train_loss_dict = json.load(open("./output/train_loss.json"))
    val_loss_dict = json.load(open("./output/val_loss.json"))

    epochs = list(map(int, train_loss_dict.keys()))
    train_loss_values = list(train_loss_dict.values())
    val_loss_values = list(val_loss_dict.values())

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss_values, label="Training Loss", marker='o')
    plt.plot(epochs, val_loss_values, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig("./output/figs/training_validation_loss_plot.png", dpi=300)
