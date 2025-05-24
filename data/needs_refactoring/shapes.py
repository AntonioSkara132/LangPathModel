import csv
import numpy as np
import svgwrite

# Function to generate points along a circle outline
def generate_circle_outline(cx, cy, r, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points)
    return [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]

# Function to generate a square outline
def generate_square_outline(x, y, size):
    return [
        (x, y), (x + size, y), (x + size, y + size), (x, y + size), (x, y)
    ]

# Define shapes with their sizes
shapes_config = {
    "small_circle": {"type": "circle", "cx": 50, "cy": 50, "r": 20, "num_points": 30},
    "circle": {"type": "circle", "cx": 150, "cy": 50, "r": 40, "num_points": 50},
    "big_circle": {"type": "circle", "cx": 250, "cy": 50, "r": 60, "num_points": 70},
    "small_square": {"type": "square", "x": 30, "y": 150, "size": 40},
    "square": {"type": "square", "x": 130, "y": 150, "size": 60},
    "big_square": {"type": "square", "x": 230, "y": 150, "size": 80}
}

# Generate shape points
shapes = {}
for shape_name, config in shapes_config.items():
    if config["type"] == "circle":
        shapes[shape_name] = generate_circle_outline(config["cx"], config["cy"], config["r"], config["num_points"])
    elif config["type"] == "square":
        shapes[shape_name] = generate_square_outline(config["x"], config["y"], config["size"])

# Save to CSV
csv_filename = "shapes_outline.csv"
with open(csv_filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Shape", "X", "Y"])
    for shape_name, points in shapes.items():
        for x, y in points:
            writer.writerow([shape_name, x, y])

print(f"✅ Shapes saved to {csv_filename}!")

# Create SVG file
svg_filename = "shapes.svg"
dwg = svgwrite.Drawing(svg_filename, profile="tiny", size=(400, 300))

# Draw shapes in SVG
for shape_name, points in shapes.items():
    dwg.add(dwg.polyline(points=points, stroke="black", fill="none"))

dwg.save()

print(f"✅ Shapes saved as SVG: {svg_filename}")


