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
    points = points.astype(float)
    points[:, 0] += disturbance_x
    points[:, 1] += disturbance_y
    return points

def change_size(points, factor):
    center = np.mean(points, axis=0)
    return (points - center) * factor + center

def calculate_time(points):
    times = [0]
    for i in range(1, len(points)):
        distance = np.linalg.norm(points[i] - points[i-1])
        times.append(times[-1] + distance)
    return np.round(np.array(times), 3)

def generate_all_shapes():
    shapes = []
    for f in f_s:
        for amplitude in sine_amplitude:
            for frequency in frequencies:
                # Circle parameters
                circle_params = {"cx": 500, "cy": 500, "r": 100, "num_points": int(np.round(f*20))}
                circle_points = generate_circle_outline(**circle_params)
                modified_circle_points = quantize(add_sinusoidal_disturbance(change_size(circle_points, f), amplitude, frequency))
                times = calculate_time(modified_circle_points)
                circle_triplets = []
                for i in range(len(modified_circle_points)):
                    circle_triplets.extend([modified_circle_points[i, 0], modified_circle_points[i, 1], times[i]])
                shapes.append(["Circle", f, frequency, amplitude, circle_params["num_points"], circle_triplets])

                # Square parameters
                square_params = {"x": 500, "y": 500, "size": 100, "num_points": int(np.round(f*20))}
                square_points = generate_square_outline(**square_params)
                modified_square_points = quantize(add_sinusoidal_disturbance(change_size(square_points, f), amplitude, frequency))
                times = calculate_time(modified_square_points)
                square_triplets = []
                for i in range(len(modified_square_points)):
                    square_triplets.extend([modified_square_points[i, 0], modified_square_points[i, 1], times[i]])
                shapes.append(["Square", f, frequency, amplitude, square_params["num_points"], square_triplets])

                # Ellipse parameters
                ellipse_params = {"cx": 500, "cy": 500, "a": 120, "b": 800, "num_points": int(np.round(f*20))}
                ellipse_points = generate_ellipse_outline(**ellipse_params)
                modified_ellipse_points = quantize(add_sinusoidal_disturbance(change_size(ellipse_points, f), amplitude, frequency))
                times = calculate_time(modified_ellipse_points)
                ellipse_triplets = []
                for i in range(len(modified_ellipse_points)):
                    ellipse_triplets.extend([modified_ellipse_points[i, 0], modified_ellipse_points[i, 1], times[i]])
                shapes.append(["Ellipse", f, frequency, amplitude, ellipse_params["num_points"], ellipse_triplets])

                # Triangle parameters
                triangle_params = {"x": 500, "y": 500, "size": 200, "num_points": int(np.round(f*20))}
                triangle_points = generate_triangle_outline(**triangle_params)
                modified_triangle_points = quantize(add_sinusoidal_disturbance(change_size(triangle_points, f), amplitude, frequency))
                times = calculate_time(modified_triangle_points)
                triangle_triplets = []
                for i in range(len(modified_triangle_points)):
                    triangle_triplets.extend([modified_triangle_points[i, 0], modified_triangle_points[i, 1], times[i]])
                shapes.append(["Triangle", f, frequency, amplitude, triangle_params["num_points"], triangle_triplets])
    return shapes

def save_to_csv(shapes, csv_filename="shapes_combined.csv"):
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Shape", "Change_Factor", "Frequency", "Sine_Amplitude", "Num_Points", "Data"])
        for shape in shapes:
            writer.writerow([shape[0], shape[1], shape[2], shape[3], shape[4]] + shape[5])

f_s = [0.25, 0.5, 1, 2, 4]  # Change size factors
sine_amplitude = [0, 2, 4, 10]  # Sine disturbance amplitudes
frequencies = [0.5, 1, 2, 4, 8, 16]  # Frequencies for sinusoidal disturbance

def main():
    print("Generating shapes...")
    shapes = generate_all_shapes()
    print("Saving shapes data to CSV...")
    save_to_csv(shapes)
    print("✅ Shapes saved to shapes_combined.csv!")

    with open("shapes_combined.csv", "r") as csvfile:
        rows = csvfile.readlines()
        print(f"Total rows in CSV (including header): {len(rows)}")
        print(f"First few rows:\n{''.join(rows[:5])}")

if __name__ == "__main__":
    main()
