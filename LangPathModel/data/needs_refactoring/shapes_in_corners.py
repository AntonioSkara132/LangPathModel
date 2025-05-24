import numpy as np
import random
import csv
import matplotlib.pyplot as plt

# Global constants
num_points = 100  # Number of points for the shape outlines
circle_center = (500, 500)  # Circle always in the middle of the map
circle_radius = 100  # Fixed radius of the circle
square_center = (500, 500)  # Square always in the middle of the map
square_size = 200  # Fixed size of the square
triangle_center = (500, 500)  # Triangle always in the middle of the map
triangle_size = 200  # Fixed size of the triangle

# Corners (lower-left, lower-right, upper-left, upper-right)
corners = [(100, 100), (900, 100), (100, 900), (900, 900)]

# Function to generate a circle outline
def generate_circle_outline(cx, cy, r, num_points=num_points):
    angles = np.linspace(0, 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles)))

# Function to generate a square outline
def generate_square_outline(x, y, size, num_points=num_points):
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

# Function to generate a triangle outline
def generate_triangle_outline(x, y, size, num_points=num_points):
    points = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        if fraction < 0.33:
            points.append([x + fraction * size, y])
        elif fraction < 0.66:
            points.append([x + size - (fraction - 0.33) * size, y + (fraction - 0.33) * size * np.sqrt(3)])
        else:
            points.append([x + (fraction - 0.66) * size, y + size * np.sqrt(3) - (fraction - 0.66) * size * np.sqrt(3)])
    return np.array(points)

# Function to find the closest point on a shape to the origin
def find_closest_point_on_shape(shape_points, start_point):
    distances = np.linalg.norm(shape_points - start_point, axis=1)  # Euclidean distance
    closest_point_idx = np.argmin(distances)
    return shape_points[closest_point_idx]

# Function to move in a straight line and generate the path to a point on the shape outline
def move_straight_line(start, shape_point):
    return np.array([start, shape_point])

# Function to add noise to the path
def add_noise(points, noise_level=0.05):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    return points + noise

# Function to write paths to CSV file
def write_paths_to_csv(filename="robot_shapes_paths.csv", num_examples=500):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        for corner in corners:
            for shape_name, shape_outline_fn, shape_size in [("Circle", generate_circle_outline, circle_radius),
                                                              ("Square", generate_square_outline, square_size),
                                                              ("Triangle", generate_triangle_outline, triangle_size)]:
                for _ in range(num_examples):  # Generate `num_examples` for each shape in each corner
                    # Write the shape and corner description
                    writer.writerow([f"draw {shape_name} in corner {corner}"])

                    # Random origin position outside the shape's boundary (outside the corner area)
                    start_point = (random.randint(0, 1000), random.randint(0, 1000))

                    # Generate the shape points
                    if shape_name == "Circle":
                        shape_points = shape_outline_fn(corner[0], corner[1], shape_size)
                    elif shape_name == "Square":
                        shape_points = shape_outline_fn(corner[0], corner[1], shape_size)
                    elif shape_name == "Triangle":
                        shape_points = shape_outline_fn(corner[0], corner[1], shape_size)

                    # Find the closest point on the shape outline
                    closest_point = find_closest_point_on_shape(shape_points, start_point)
                    
                    # Generate the straight line path from the start to the closest point
                    path_to_shape = move_straight_line(start_point, closest_point)
                    
                    # Prepare the path data (moving towards the shape without drawing)
                    path_data = []
                    
                    # Moving towards the shape without drawing (action = 0)
                    for i in range(len(path_to_shape) - 1):
                        x1, y1 = path_to_shape[i]
                        x2, y2 = path_to_shape[i + 1]
                        path_data.extend([x1, y1, 0, x2, y2, 0])  # Action = 0 (not drawing)
                    
                    # Now draw the shape (action = 1)
                    for i in range(len(shape_points) - 1):
                        x1, y1 = shape_points[i]
                        x2, y2 = shape_points[i + 1]
                        path_data.extend([x1, y1, 1, x2, y2, 1])  # Action = 1 (drawing)
                    
                    # Add noise to the path data
                    path_data = np.round(add_noise(np.array(path_data).reshape(-1, 6)))  # Reshape for correct dimension
                    
                    # Write the path data to CSV
                    writer.writerow(path_data.flatten())

    print(f"Paths written to {filename}")

# Main function to generate and save paths to CSV
def main():
    print("Generating paths and saving them to CSV...")
    write_paths_to_csv(filename="robot_shapes_paths.csv", num_examples=500)
    print("✅ Paths saved to robot_shapes_paths.csv!")

if __name__ == "__main__":
    main()
