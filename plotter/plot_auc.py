import matplotlib.pyplot as plt
import json

def make_auc_auprc():
    file = open("./output/metric_history.json")
    file_dict = json.load(file)

    epochs = [item["epoch"] for item in file_dict]
    auc = [item["auc"] for item in file_dict]
    auprc = [item["auprc"] for item in file_dict]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, auc, label="AUC", marker='o')
    plt.plot(epochs, auprc, label="AUPRC", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("AUC and AUPRC over Epochs")
    plt.legend()
    plt.grid(True)


    plt.savefig("./output/figs/auc_auprc_plot.png", dpi=300)
    plt.close()