import numpy as np
import matplotlib.pyplot as plt
import os

def make_confusion_matrix_grid():
    # Load confusion matrices
    conf_mats = np.load('./output/conf_mat.npy')

    # Ensure the output directory exists
    os.makedirs('./output/figs', exist_ok=True)

    # Calculate rows and columns
    ncols = 4
    nrows = (len(conf_mats) + ncols - 1) // ncols  # Round up to fit all matrices


    # Create figure and axes
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
    fig.suptitle('Confusion Matrix Grid', fontsize=16, y=0.95)

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Plot each confusion matrix
    for i, conf_mat in enumerate(conf_mats):
        ax = axes[i]

        # Plot confusion matrix
        im = ax.imshow(conf_mat, cmap='coolwarm', interpolation='nearest')
        ax.set_title(f'Confusion Matrix {i + 1}', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count', fontsize=10)

        # Add labels and ticks
        ax.set_xticks(range(conf_mat.shape[1]))
        ax.set_yticks(range(conf_mat.shape[0]))
        ax.set_xticklabels(['True', 'False'], fontsize=10)
        ax.set_yticklabels(['True', 'False'], fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)

    # Hide unused subplots
    for j in range(len(conf_mats), len(axes)):
        axes[j].axis('off')

    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    plt.savefig("./output/figs/confusion_matrix_grid.png", dpi=300)
    plt.close()
