import numpy as np
import random
import csv
import matplotlib.pyplot as plt

# Global constants
M = 1000  # Quantization factor (you can adjust as needed)
num_points = 100  # Number of points for square outline
square_center = (500, 500)  # Square always in the middle of the map
square_size = 200  # Fixed size of the square

# Function to generate a square outline
def generate_square_outline(x, y, size, num_points=num_points):
    # Generate the points for a square's outline
    points = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        if fraction < 0.25:
            points.append([x + fraction * size, y])
        elif fraction < 0.5:
            points.append([x + size, y + (fraction - 0.25) * size])
        elif fraction < 0.75:
            points.append([x + size - (fraction - 0.5) * size, y + size])
        else:
            points.append([x, y + size - (fraction - 0.75) * size])
    return np.array(points)

# Function to move in a straight line and generate the path to a point on the square outline
def move_straight_line(start, square_point):
    return np.array([start, square_point])

# Function to find the closest point on the square to the origin (start point)
def find_closest_point_on_square(x, y, size, start_point):
    # Generate the full square outline
    square_points = generate_square_outline(x, y, size)
    
    # Calculate the distance from the start point to each point on the square's outline
    distances = np.linalg.norm(square_points - start_point, axis=1)  # Euclidean distance
    
    # Find the index of the closest point
    closest_point_idx = np.argmin(distances)
    
    # Return the closest point
    return square_points[closest_point_idx]

# Function to add noise to the path
def add_noise(points, noise_level=0.05):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    return points + noise

# Function to plot squares and paths
def plot_squares_with_paths(num_origins=5):
    plt.figure(figsize=(6, 6))
    
    for _ in range(num_origins):
        # Random origin position outside the square's boundary
        start_point = (random.randint(0, 1000), random.randint(0, 1000))
        
        # Find the closest point on the square's outline to the origin
        closest_point = find_closest_point_on_square(square_center[0], square_center[1], square_size, start_point)
        
        # Generate the straight line path from the start to the closest point on the square
        path_to_square = move_straight_line(start_point, closest_point)
        
        # Plot the straight line path and the square
        plt.plot(path_to_square[:, 0], path_to_square[:, 1], '--', label=f"Path from {start_point} to closest point")
        square_points = generate_square_outline(square_center[0], square_center[1], square_size)
        plt.plot(square_points[:, 0], square_points[:, 1], label="Square with fixed center (500, 500) and size=200")
    
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.title("Paths to Closest Points on Square and Square Drawn from Random Origins")
    #plt.xlabel("X coordinate")
    #plt.ylabel("Y coordinate")
    #plt.legend()
    #plt.grid(True)
    #plt.show()

# Function to write paths to CSV file
def write_paths_to_csv(num_origins=5, filename="robot_square_paths.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x1", "y1", "a1", "x2", "y2", "a2", "x3", "y3", "a3", ...])  # Include as many columns as needed
        
        for _ in range(num_origins):
            # Random origin position outside the square's boundary
            start_point = (random.randint(0, 1000), random.randint(0, 1000))
            
            # Find the closest point on the square's outline to the origin
            closest_point = find_closest_point_on_square(square_center[0], square_center[1], square_size, start_point)
            
            # Generate the straight line path from the start to the closest point on the square
            path_to_square = move_straight_line(start_point, closest_point)
            
            # Prepare the triplets for writing to CSV (moving towards the square without drawing)
            path_data = []
            
            # Moving towards the square without drawing
            for i in range(len(path_to_square) - 1):
                x1, y1 = path_to_square[i]
                x2, y2 = path_to_square[i + 1]
                path_data.extend([x1, y1, 0, x2, y2, 0])  # Action = 0 (not drawing)
            
            # Now draw the square (drawing action = 1)
            square_points = generate_square_outline(square_center[0], square_center[1], square_size)
            
            for i in range(len(square_points) - 1):
                x1, y1 = square_points[i]
                x2, y2 = square_points[i + 1]
                path_data.extend([x1, y1, 1, x2, y2, 1])  # Action = 1 (drawing)
            
            # Add noise to the path data
            path_data = add_noise(np.array(path_data).reshape(-1, 6))  # Reshape for correct dimension
            # Write the data to CSV
            writer.writerow(path_data.flatten())

    print(f"Paths written to {filename}")

# Main function to generate and save paths to CSV
def main():
    print("Generating paths and saving them to CSV...")
    write_paths_to_csv(num_origins=1000, filename="square_in_the_middle.csv")
    print("✅ Paths saved to square_in_the_middle.csv!")

if __name__ == "__main__":
    main()
