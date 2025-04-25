import numpy as np
import random
import torch
import matplotlib.pyplot as plt

# Constants
M = 1000
num_points_per_side = 20  # total 4 sides * 20 = 80 points
square_center = (500, 200)
square_size = 200

def generate_square_outline(x, y, size):
    half = size / 2
    # Define 4 corners
    top_left = (x - half, y - half)
    top_right = (x + half, y - half)
    bottom_right = (x + half, y + half)
    bottom_left = (x - half, y + half)
    
    # Interpolate points along each edge
    top = np.linspace(top_left, top_right, num_points_per_side)
    right = np.linspace(top_right, bottom_right, num_points_per_side)
    bottom = np.linspace(bottom_right, bottom_left, num_points_per_side)
    left = np.linspace(bottom_left, top_left, num_points_per_side)
    
    return np.vstack([top, right, bottom, left])

def move_straight_line(start, end, step_size=50):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    distance = np.linalg.norm(vector)

    if distance == 0:
        return np.array([start])

    direction = vector / distance
    num_steps = int(distance // step_size)

    points = [start + i * step_size * direction for i in range(num_steps + 1)]
    points.append(end)
    return np.array(points)

def find_closest_point_on_square(square_points, start_point):
    distances = np.linalg.norm(square_points - start_point, axis=1)
    idx = np.argmin(distances)
    return square_points[idx]

def generate_square_starting_at_point(square_points, start_point):
    distances = np.linalg.norm(square_points - start_point, axis=1)
    start_idx = np.argmin(distances)
    # Roll array so it starts at closest point
    rolled = np.roll(square_points, -start_idx, axis=0)
    return rolled

def add_noise(points, noise_level=0.05):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    return points + noise

def write_square_paths_to_pt(num_origins=5, filename="square_paths.pt"):
    all_data = []

    for _ in range(num_origins):
        start_point = (random.randint(0, 1000), random.randint(0, 1000))
        square_points = generate_square_outline(square_center[0], square_center[1], square_size)
        closest_point = find_closest_point_on_square(square_points, start_point)
        path_to_square = move_straight_line(start_point, closest_point)

        path_data = []
        for x, y in path_to_square:
            path_data.append([x, y, 0, 0])  # move

        ordered_square_points = generate_square_starting_at_point(square_points, closest_point)
        for i in range(len(ordered_square_points) - 1):
            x1, y1 = ordered_square_points[i]
            x2, y2 = ordered_square_points[i + 1]
            path_data.append([x1, y1, 1, 0])
            path_data.append([x2, y2, 1, 0])

        path_data[-1][-1] = 1  # stop = 1

        path_tensor = torch.tensor(path_data, dtype=torch.float32)
        path_tensor[:, 0:2] = add_noise(path_tensor[:, 0:2])

        all_data.append({
            "path": path_tensor,
            "text": "square on the bottom side"
        })

    torch.save(all_data, filename)
    print(f"{len(all_data)} square paths saved to {filename}")

def visualize_square_paths_from_file(filename="square_paths.pt", num_paths=5):
    data = torch.load(filename)
    plt.figure(figsize=(6, 6))

    for i in range(min(num_paths, len(data))):
        path = data[i]['path']
        x = path[:, 0].numpy()
        y = path[:, 1].numpy()
        actions = path[:, 2].numpy()

        draw = actions > 0.5
        move = actions <= 0.5

        plt.scatter(x[move], y[move], c='blue', marker='o', label='move' if i == 0 else "")
        plt.scatter(x[draw], y[draw], c='red', marker='x', label='draw' if i == 0 else "")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Generated Square Paths with Actions")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    write_square_paths_to_pt(num_origins=2000, filename="square_paths_bottom.pt")
    visualize_square_paths_from_file("square_paths_bottom.pt", num_paths=5)

if __name__ == "__main__":
    main()

