import torch
from data_utils import PathDataset
import sys
import numpy as np
from collections import defaultdict

"""
Script that computes and prints dataset statistics from a .pt file
"""

def compute_path_length(path_np):
    diffs = np.diff(path_np[:, :2], axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return np.sum(dists)

if __name__ == "__main__":
    data_file = sys.argv[1]
    dataset = PathDataset(data_file)

    total_paths = len(dataset)
    total_points = 0
    total_length = 0.0
    label_counts = defaultdict(int)
    label_point_counts = defaultdict(list)
    label_lengths = defaultdict(list)

    for i in range(total_paths):
        path, text = dataset[i]
        path_np = path.numpy()

        num_points = len(path_np)
        path_len = compute_path_length(path_np)

        total_points += num_points
        total_length += path_len
        label_counts[text] += 1
        label_point_counts[text].append(num_points)
        label_lengths[text].append(path_len)

    unique_labels = list(label_counts.keys())
    avg_points_per_path = total_points / total_paths
    avg_path_length = total_length / total_paths

    print(f"Dataset: {data_file}")
    print(f"Total paths: {total_paths}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Average points per path: {avg_points_per_path:.2f}")
    print(f"Average path length: {avg_path_length:.4f}")

    print("\n Paths per label:")
    for label in sorted(label_counts, key=label_counts.get, reverse=True):
        count = label_counts[label]
        mean_pts = np.mean(label_point_counts[label])
        mean_len = np.mean(label_lengths[label])
        print(f"  - '{label}': {count} paths | Avg points: {mean_pts:.1f} | Avg length: {mean_len:.4f}")
