import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def calculate_fpr_tpr(conf_mat):
    # Extract values from confusion matrix
    TP = conf_mat[1, 1]
    FP = conf_mat[0, 1]
    TN = conf_mat[0, 0]
    FN = conf_mat[1, 0]

    # Calculate FPR and TPR
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return FPR, TPR

def make_roc():
    conf_mats = np.load('./output/conf_mat.npy')
    n_classes = conf_mats.shape[0]
    fprs, tprs = [], []

    for mat in conf_mats:
        fpr, tpr = calculate_fpr_tpr(mat)

        fprs.append(fpr)
        tprs.append(tpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fprs, tprs, label="ROC Curve", color="blue")
    plt.plot([0, 1], [0, 1], 'k--', label="Chance")  # Diagonal line
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("./output/figs/roc_plot.png", dpi=300)
