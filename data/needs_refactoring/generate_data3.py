import numpy as np
import random
import csv
import matplotlib.pyplot as plt

# Global constants
M = 1000  # Quantization factor (you can adjust as needed)
num_points = 100  # Number of points for circle outline
circle_center = (500, 500)  # Circle always in the middle of the map
circle_radius = 100  # Fixed radius of the circle

# Function to generate a circle outline
def generate_circle_outline(cx, cy, r, num_points=num_points):
    start_phase_t = np.random.uniform(0, 2 * np.pi)
    angles = np.linspace(start_phase_t, start_phase_t + 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))

# Function to move in a straight line and generate path to a point on the circle outline
def move_straight_line(start, circle_point):
    return np.array([start, circle_point])

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
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Paths to Closest Points on Circle and Circle Drawn from Random Origins")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to write paths to CSV file
def write_paths_to_csv(num_origins=5, filename="robot_paths.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "y1", "a1", "x2", "y2", "a2", "x3", "y3", "a3", ...])  # Include as many columns as needed
        
        for _ in range(num_origins):
            # Random origin position outside the circle's boundary
            start_point = (random.randint(0, 1000), random.randint(0, 1000))
            
            # Find the closest point on the circle's outline to the origin
            closest_point = find_closest_point_on_circle(circle_center[0], circle_center[1], circle_radius, start_point)
            
            # Generate the straight line path from the start to the closest point on the circle
            path_to_circle = move_straight_line(start_point, closest_point)
            
            # Prepare the triplets for writing to CSV (moving towards the circle without drawing)
            path_data = []
            
            # Moving towards the circle without drawing
            for i in range(len(path_to_circle) - 1):
                x1, y1 = path_to_circle[i]
                x2, y2 = path_to_circle[i + 1]
                path_data.extend([x1, y1, 0, x2, y2, 0])  # Action = 0 (not drawing)
            
            # Now draw the circle (drawing action = 1)
            circle_points = generate_circle_outline(circle_center[0], circle_center[1], circle_radius)
            
            for i in range(len(circle_points) - 1):
                x1, y1 = circle_points[i]
                x2, y2 = circle_points[i + 1]
                path_data.extend([x1, y1, 1, x2, y2, 1])  # Action = 1 (drawing)
            
            # Write the data to CSV
            writer.writerow(path_data)

    print(f"Paths written to {filename}")

# Main function to generate and save paths to CSV
def main():
    print("Generating paths and saving them to CSV...")
    write_paths_to_csv(num_origins=5, filename="circle_in_the middle.csv")
    print("✅ Paths saved to circle_in_the_middle.csv!")

if __name__ == "__main__":
    main()
