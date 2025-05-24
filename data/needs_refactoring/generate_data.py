import numpy as np
import csv
import svgwrite

M = 1000  # Global constant for quantization

def quantize(points):
    return np.round(points).astype(int)

def generate_circle_outline(cx, cy, r, num_points=20):
    angles = np.linspace(0, 2 * np.pi, num_points)
    return np.column_stack((cx + r * np.cos(angles), cy + r * np.sin(angles))).astype(float)

def generate_square_outline(x, y, size):
    return np.array([
        [x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]
    ]).astype(float)

def generate_ellipse_outline(cx, cy, a, b, num_points=20):
    angles = np.linspace(0, 2 * np.pi, num_points)
    return np.column_stack((cx + a * np.cos(angles), cy + b * np.sin(angles))).astype(float)

def add_noise(points, noise_level=0.5):
    noise = np.random.normal(scale=noise_level, size=points.shape)
    return points + noise

def add_sinusoidal_disturbance(points, amplitude=2.0, f=4.0):
    num_points = points.shape[0]
    start_phase_x = np.random.uniform(0, 2 * np.pi)
    start_phase_y = np.random.uniform(0, 2 * np.pi)
    disturbance_x = amplitude * np.sin(np.linspace(start_phase_x, start_phase_x + f * 2 * np.pi, num_points))
    disturbance_y = amplitude * np.sin(np.linspace(start_phase_y, start_phase_y + f * 2 * np.pi, num_points))
    points = points.astype(float)  # Ensure float operations
    points[:, 0] += disturbance_x  # Apply disturbance to x-coordinates
    points[:, 1] += disturbance_y  # Apply disturbance to y-coordinates
    return points

def change_size(points, factor):
    center = np.mean(points, axis=0)
    return (points - center) * factor + center

def save_as_svg(shapes, filename="shapes_sine0_fac1.svg", size=(M, M)):
    dwg = svgwrite.Drawing(filename, size=size, profile='tiny')
    for shape_name, points in shapes.items():
        centered_points = points + np.array([size[0] / 2, size[1] / 2])
        dwg.add(dwg.polyline(centered_points, stroke='black', fill='none'))
    dwg.save()
    print(f"✅ Shapes saved as SVG to {filename}!")

def calculate_time(points):
    # Calculate Euclidean distance between consecutive points and accumulate the time
    times = [0]  # Time starts at 0
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i-1])
        times.append(times[-1] + distance)
    return np.array(times)


def save_as_csv2(shapes, filename):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Shape", "X", "Y"])
        for shape_name, points in shapes.items():
            for x, y in points:
                writer.writerow([shape_name, x, y])

shapes = {
    "Circle": generate_circle_outline(0, 0, 100),
    "Square": generate_square_outline(0, 0, 100),
    "Ellipse": generate_ellipse_outline(0, 0, 150, 100)
}

# Apply noise and sinusoidal disturbance, then quantize
#noisy_shapes = {name: quantize(add_sinusoidal_disturbance(points)) for name, points in shapes.items()}
"""
noisy_shapes = {name: quantize(change_size(add_sinusoidal_disturbance(points, amplitude=0.0), 1)) for name, points in shapes.items()}

csv_filename = "shapes_sine3_fac1.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Shape", "X", "Y"])
    for shape_name, points in noisy_shapes.items():
        for x, y in points:
            writer.writerow([shape_name, x, y])

print(f"✅ Shapes with noise and sinusoidal disturbance saved to {csv_filename}!")

# Save as SVG
save_as_svg(noisy_shapes)
"""
"""

# Parameter grids
f_s = [0.25, 0.5, 1, 2, 4]  # Change size factors
sine_amplitude = [0, 2, 4, 10]  # Sine disturbance amplitudes
frequencies = [0.5, 1, 2, 4, 8, 16]  # Frequencies for sinusoidal disturbance

shapes = {
    "Circle": generate_circle_outline(500, 500, 100),
    "Square": generate_square_outline(500, 500, 100),
    "Ellipse": generate_ellipse_outline(500, 500, 150, 100)
}

"""
def save_to_csv(shapes, f_s, sine_amplitude, frequencies, num_points_grid, csv_filename="shapes_combined.csv"):
    with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row with shape name, change_factor, frequency, and sine_amplitude
            writer.writerow(["Shape", "Change_Factor", "Frequency", "Sine_Amplitude", "Num_Points", "Time", "Coordinates"])

            # Iterate over all shapes and parameter combinations
            for shape_name, points in shapes.items():
                for f in f_s:
                    for amplitude in sine_amplitude:
                        for frequency in frequencies:
                            for num_points in num_points_grid:
                                # Generate the shape with the current num_points
                                modified_points = quantize(add_sinusoidal_disturbance(change_size(points, f), amplitude, frequency))

                                # Adjust shape to the new number of points
                                if num_points != points.shape[0]:
                                    modified_points = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, points.shape[0]), modified_points.flatten()).reshape(-1, 2)

                                # Apply quantization and transformations
                                modified_points = quantize(modified_points)
                            
                                # Calculate the time (cumulative Euclidean distance)
                                times = calculate_time(modified_points)

                                # Flatten the points and time into a single list [time1, x1, y1, time2, x2, y2, ...]
                                flattened_data = times.tolist() + modified_points.flatten().tolist()

                                # Write the shape name, parameters, num_points, followed by time and coordinates
                                writer.writerow([shape_name, f, frequency, amplitude, num_points] + flattened_data)

# Parameter grids
"""
f_s = [0.25, 0.5, 1, 2, 4]  # Change size factors
sine_amplitude = [0, 2, 4, 10]  # Sine disturbance amplitudes
frequencies = [0.5, 1, 2, 4, 8, 16]  # Frequencies for sinusoidal disturbance

shapes = {
    "Circle": generate_circle_outline(500, 500, 100),
    "Square": generate_square_outline(500, 500, 100),
    "Ellipse": generate_ellipse_outline(500, 500, 150, 100)
}

# Save all the data in the same CSV file
save_to_csv(shapes)

print("✅ All shapes saved to shapes_combined.csv!")
"""