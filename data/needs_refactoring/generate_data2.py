import numpy as np
import csv
import svgwrite

M = 1000  # Global constant for quantization

def quantize(points):
    return np.round(points).astype(int)

def generate_circle_outline(cx, cy, r, num_points=100):
    start_phase_t = np.random.uniform(0, 2 * np.pi)
    angles = np.linspace(start_phase_t, start_phase_t + 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles))).astype(float)

def generate_square_outline(x, y, size, num_points=100):
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
    return np.array(points).astype(float)

def generate_ellipse_outline(cx, cy, a, b, num_points=100):
    start_phase_t = np.random.uniform(0, 2 * np.pi)
    angles = np.linspace(start_phase_t, start_phase_t + 2 * np.pi, num_points)
    return np.column_stack((cx + a * np.cos(angles), cy + b * np.sin(angles))).astype(float)

def generate_triangle_outline(x, y, size, num_points=100):
    points = []
    for i in range(num_points):
        fraction = i / (num_points - 1)
        if fraction < 0.33:
            points.append([x + fraction * size, y])
        elif fraction < 0.66:
            points.append([x + size - (fraction - 0.33) * size, y + (fraction - 0.33) * size * np.sqrt(3)])
        else:
            points.append([x + (fraction - 0.66) * size, y + size * np.sqrt(3) - (fraction - 0.66) * size * np.sqrt(3)])
    return np.array(points).astype(float)

def add_noise(points, noise_level=0.05):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    return points + noise

def add_sinusoidal_disturbance(points, amplitude=5.0, frequency=1.0):
    num_points = points.shape[0]
    start_phase_x = np.random.uniform(0, 2 * np.pi)
    start_phase_y = np.random.uniform(0, 2 * np.pi)
    disturbance_x = amplitude * np.sin(frequency * np.linspace(start_phase_x, start_phase_x + 2 * np.pi, num_points))
    disturbance_y = amplitude * np.sin(frequency * np.linspace(start_phase_y, start_phase_y + 2 * np.pi, num_points))
    points = points.astype(float)  # Ensure float operations
    points[:, 0] += disturbance_x  # Apply disturbance to x-coordinates
    points[:, 1] += disturbance_y  # Apply disturbance to y-coordinates
    return points

def change_size(points, factor):
    center = np.mean(points, axis=0)
    return (points - center) * factor + center

def calculate_time(points):
    # Calculate Euclidean distance between consecutive points and accumulate the time
    times = [0]  # Time starts at 0
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i-1])
        times.append(times[-1] + distance)
    # Round times to 3 decimal places
    return np.round(np.array(times), 3)

def save_to_csv(shapes, csv_filename="shapes_combined.csv"):
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row with shape name, change_factor, frequency, and sine_amplitude
        writer.writerow(["Shape", "Change_Factor", "Frequency", "Sine_Amplitude", "Num_Points", "Data"])

        # Iterate over all shapes and parameter combinations
        for shape_name, points in shapes.items():
            for f in f_s:
                for amplitude in sine_amplitude:
                    for frequency in frequencies:
                        # Apply transformations and disturbances
                        modified_points = quantize(add_sinusoidal_disturbance(change_size(points, f), amplitude, frequency))

                        # Calculate the time (cumulative Euclidean distance)
                        times = calculate_time(modified_points)

                        # Organize the data into triplets [x1, y1, t1, x2, y2, t2, ...]
                        triplets = []
                        for i in range(len(modified_points)):
                            triplets.extend([modified_points[i, 0], modified_points[i, 1], times[i]])

                        # Write the shape name, parameters, num_points, followed by the triplets
                        writer.writerow([shape_name, f, frequency, amplitude, points.shape[0]] + triplets)

# Parameter grids
f_s = [0.25, 0.5, 1, 2, 4]  # Change size factors
sine_amplitude = [0, 2, 4, 10]  # Sine disturbance amplitudes
frequencies = [0.5, 1, 2, 4, 8, 16]  # Frequencies for sinusoidal disturbance

shapes = {
    "Circle": generate_circle_outline(500, 500, 100, num_points=20),
    "Square": generate_square_outline(500, 500, 100, num_points=10),
    "Ellipse": generate_ellipse_outline(500, 500, 150, 100, num_points=20),
    "Triangle": generate_triangle_outline(500, 500, 100, num_points=20)  # Adding triangle shape
}

def main():
    # Test 1: Check if CSV is generated properly
    print("Saving shapes data to CSV...")
    save_to_csv(shapes)
    print("✅ Shapes saved to shapes_combined.csv!")

    # Test 2: Check if we have the expected number of rows in CSV
    with open("shapes_combined.csv", "r") as csvfile:
        rows = csvfile.readlines()
        print(f"Total rows in CSV (including header): {len(rows)}")
        print(f"First few rows:\n{''.join(rows[:5])}")  # Displaying first few rows

    # Test 3: Check if the shapes dictionary contains all shapes
    print("\nShape Names and Their Points:")
    for shape_name, points in shapes.items():
        print(f"{shape_name}: {points.shape[0]} points")

if __name__ == "__main__":
    main()
