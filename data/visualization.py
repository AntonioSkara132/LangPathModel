import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Function to read CSV and extract data
def read_csv(csv_filename="shapes_combined.csv"):
    shapes_data = {}
    
    with open(csv_filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header row
        
        for row in reader:
            shape_name = row[0]
            change_factor = float(row[1])
            frequency = float(row[2])
            sine_amplitude = float(row[3])
            num_points = int(row[4])
            
            # Extract coordinates and times
            points = np.array([list(map(float, row[i:i+3])) for i in range(5, len(row), 3)])
            x = points[:, 0]
            y = points[:, 1]
            t = points[:, 2]  # Time values
            
            # Store the shape data
            shape_key = f"{shape_name}_{change_factor}_{frequency}_{sine_amplitude}_{num_points}"
            shapes_data[shape_key] = (x, y, t)
    
    return shapes_data

# Function to visualize data
def visualize_data(shapes_data):
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Set a colormap for time visualization
    colormap = cm.viridis  # You can change this to another colormap if desired
    
    for shape_key, (x, y, t) in shapes_data.items():
        # Normalize time (t) for color mapping
        norm = plt.Normalize(vmin=t.min(), vmax=t.max())
        sc = plt.scatter(x, y, c=t, cmap=colormap, norm=norm, label=shape_key, s=30)

    # Add a colorbar
    plt.colorbar(sc, label="Time (t)")

    # Add labels and title
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Visualization of Shapes with Time (t) as Color")
    plt.legend(loc="upper right")
    
    # Show the plot
    plt.show()

def main():
    # Read the CSV data
    shapes_data = read_csv("shapes_combined.csv")
    
    # Visualize the data
    visualize_data(shapes_data)

if __name__ == "__main__":
    main()
