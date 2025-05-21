from generate_data import save_to_csv, generate_circle_outline, generate_square_outline, generate_ellipse_outline


# Parameter grids
f_s = [0.25, 0.5, 1, 2, 4]  # Change size factors
sine_amplitude = [0, 2, 4, 10]  # Sine disturbance amplitudes
frequencies = [0.5, 1, 2, 4, 8, 16]  # Frequencies for sinusoidal disturbance
num_points_grid = [10, 20, 40, 80]


shapes = {
    "Circle": generate_circle_outline(500, 500, 100),
    "Square": generate_square_outline(500, 500, 100),
    "Ellipse": generate_ellipse_outline(500, 500, 150, 100)
}

# Save all the data in the same CSV file
save_to_csv(shapes, f_s, sine_amplitude, frequencies, num_points_grid)

print("✅ All shapes saved to shapes_combined.csv!")