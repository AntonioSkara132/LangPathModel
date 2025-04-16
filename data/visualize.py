import matplotlib.pyplot as plt
import torch

data = torch.load('/home/antonio/Workspace/Seminar/LangPathModel/data/circle_in_the middle.pt')
plt.figure(figsize=(6, 6))

for i in range(min(5, len(data))):
    path = data[i]['path']
    x = path[:, 0].numpy()
    y = path[:, 1].numpy()
    actions = path[:, 2].numpy()  # 0 = move, 1 = draw

    # Scatter plot for each action type
    move_indices = actions == 0
    draw_indices = actions == 1

    plt.scatter(x[move_indices], y[move_indices], c='blue', marker='o', label=f'Path {i+1} - move' if i == 0 else "")
    plt.scatter(x[draw_indices], y[draw_indices], c='orange', marker='x', label=f'Path {i+1} - draw' if i == 0 else "")

plt.gca().set_aspect('equal', adjustable='box')
plt.title(f"Sample Paths with Actions (first {5})")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()