import torch
from data_utils import PathDataset
import sys
import matplotlib.pyplot as plt
import numpy as np

"""
File that visualizes generated data from .pt file
"""

if __name__ == "__main__":
    data_file = sys.argv[1]
    dataset = PathDataset(data_file)

    num_to_plot = 12
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14))  # Square plots

    for i in range(num_to_plot):
        index = np.random.randint(0, len(dataset))
        path, text = dataset[index]
        path_np = path.numpy()

        x, y = path_np[:, 0], path_np[:, 1]

        actions = path_np[:, 2]
        time = np.linspace(0, 1, len(x))

        ax = axes[i // cols, i % cols]

        move_mask = actions == 0
        draw_mask = actions == 1

        move = ax.scatter(x[move_mask], y[move_mask], c=time[move_mask], cmap='Blues', marker='o', s=20, label="Move")
        draw = ax.scatter(x[draw_mask], y[draw_mask], c=time[draw_mask], cmap='Oranges', marker='x', s=20, label="Draw")

        ax.plot(x, y, linewidth=4, alpha=0.2, color='gray')

        ax.set_title(f"Sample {i + 1}: \"{text}\"", fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True)

        # Add colorbar to each subplot
        cbar = fig.colorbar(draw, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Time Progression")

    plt.tight_layout()
    plt.show()


