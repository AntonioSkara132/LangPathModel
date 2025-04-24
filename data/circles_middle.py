import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import torch

# Global constants
M = 1000  # Quantization factor (you can adjust as needed)
num_points = 40  # Number of points for circle outline
circle_center = (500, 500)  # Circle always in the middle of the map
circle_radius = 200  # Fixed radius of the circle

# Function to generate a circle outline
def generate_circle_outline(cx, cy, r, num_points=num_points):
    start_phase_t = np.random.uniform(0, 2 * np.pi)
    angles = np.linspace(start_phase_t, start_phase_t + 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))
    
def generate_circle_starting_at_point(cx, cy, r, start_point, num_points=num_points):
    # Compute angle between center and start_point
    dx, dy = start_point[0] - cx, start_point[1] - cy
    start_angle = np.arctan2(dy, dx)

    # Generate angles starting at the correct angle
    angles = np.linspace(start_angle, start_angle + 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))

# Function to move in a straight line and generate path to a point on the circle outline
def move_straight_line(start, end, step_size=50):
    start = np.array(start)
    end = np.array(end)
    vector = end - start
    distance = np.linalg.norm(vector)

    if distance == 0:
        return np.array([start])  # No movement

    direction = vector / distance
    num_steps = int(distance // step_size)

    points = [start + i * step_size * direction for i in range(num_steps + 1)]
    points.append(end)  # ensure we include the exact closest point as the last point
    points = np.array(points)
    return points
# Function to find the closest point on the circle to the origin (start point)
def find_closest_point_on_circle(cx, cy, r, start_point):
    # Generate the full circle outline
    circle_points = generate_circle_outline(cx, cy, r)

    # Calculate the distance from the start point to each point on the circle's outline
    distances = np.linalg.norm(circle_points - start_point, axis=1)  # Euclidean distance
    
    # Find the index of the closest point
    closest_point_idx = np.argmin(distances)
    
    # Return the closest point
    return circle_points[closest_point_idx]

# Function to plot the circle and the straight path to the closest point on the circle outline

def add_noise(points, noise_level=0.05):
    noise = np.random.normal(scale=noise_level, size=1)
    return points + noise

def plot_circles_with_paths(num_origins=5):
    plt.figure(figsize=(6, 6))
    
    for _ in range(num_origins):
        # Random origin position outside the circle's boundary
        start_point = (random.randint(0, 1000), random.randint(0, 1000))
        
        # Find the closest point on the circle's outline to the origin
        closest_point = find_closest_point_on_circle(circle_center[0], circle_center[1], circle_radius, start_point)
        
        # Generate the straight line path from the start to the closest point on the circle
        path_to_circle = move_straight_line(start_point, closest_point)
        
        # Plot the straight line path and the circle
        plt.plot(path_to_circle[:, 0], path_to_circle[:, 1], '--', label=f"Path from {start_point} to closest point")
        circle_points = generate_circle_outline(circle_center[0], circle_center[1], circle_radius)
        plt.plot(circle_points[:, 0], circle_points[:, 1], label="Circle with fixed center (500, 500) and r=100")
    
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.title("Paths to Closest Points on Circle and Circle Drawn from Random Origins")
    #plt.xlabel("X coordinate")
    #plt.ylabel("Y coordinate")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

# Function to write paths to CSV file
def write_paths_to_pt(num_origins=5, filename="robot_paths.pt"):
    circle_center = (500, 500)
    circle_radius = 100
    all_data = []

    for _ in range(num_origins):
        start_point = (random.randint(0, 1000), random.randint(0, 1000))
        closest_point = find_closest_point_on_circle(circle_center[0], circle_center[1], circle_radius, start_point)
        path_to_circle = move_straight_line(start_point, closest_point)

        path_data = []
        for x, y in path_to_circle:
            path_data.append([x, y, 0, 0])  # a = 0, stop = 0

        circle_points = generate_circle_starting_at_point(circle_center[0], circle_center[1], circle_radius, closest_point)

        for i in range(len(circle_points) - 1):
            x1, y1 = circle_points[i]
            x2, y2 = circle_points[i + 1]
            path_data.append([x1, y1, 1, 0])
            path_data.append([x2, y2, 1, 0])

        # set stop flag for last point
        path_data[-1][-1] = 1  # stop = 1

        path_tensor = torch.tensor(path_data, dtype=torch.float32)
        path_tensor[:, 0:2] = add_noise(path_tensor)[:, 0:2]

        all_data.append({
            "path": path_tensor,
            "text": "circle"
        })

    torch.save(all_data, filename)
    print(f"{len(all_data)} annotated paths saved to {filename}")


    torch.save(all_data, filename)

    # Save list of tensors to a file
    torch.save(all_data, filename)
    print(f"{len(all_data)} paths saved to {filename}")
def visualize_paths_from_file(filename="circle_in_the middle.pt", num_paths=5):
    import matplotlib.pyplot as plt
    import torch

    data = torch.load(filename)
    plt.figure(figsize=(6, 6))

    for i in range(min(num_paths, len(data))):
        path = data[i]['path']
        x = path[:, 0].numpy()
        y = path[:, 1].numpy()
        actions = path[:, 2].numpy()  # 0 = move, 1 = draw

        # Scatter plot for each action type
        move_indices = actions > 0.5
        draw_indices = actions < 0.5

        plt.scatter(x[move_indices], y[move_indices], c='blue', marker='o', label=f'Path {i+1} - move' if i == 0 else "")
        plt.scatter(x[draw_indices], y[draw_indices], c='orange', marker='x', label=f'Path {i+1} - draw' if i == 0 else "")

    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Sample Paths with Actions (first {num_paths})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to generate and save paths to CSV
def main():
    print("Generating paths and saving them to CSV...")
    write_paths_to_pt(num_origins=10000, filename="circle_in_the middle.pt")
    print("✅ Paths saved to circle_in_the_middle.csv!")
    visualize_paths_from_file("circle_in_the middle.pt", num_paths=5)

if __name__ == "__main__":
    main()
