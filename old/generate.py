import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
output_dir = "generated_numbers"
os.makedirs(output_dir, exist_ok=True)

# Font properties
font = {'family': 'Calibri', 'size': 150}  # Update 'family' as needed

# Generate images for numbers 0-9
for number in range(10):
    for rotation in range(0, 360, 45):  # Rotate in 45-degree increments
        fig, ax = plt.subplots()

        # Text properties
        text = ax.text(0.5, 0.5, str(number), fontdict=font, ha='center', va='center', rotation=rotation)

        # Hide axes
        ax.axis('off')

        # Set figure background color
        fig.patch.set_facecolor('white')

        # Save the figure
        filename = os.path.join(output_dir, f"{number}_rotation_{rotation}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=100)

        # Close the plot to free memory
        plt.close(fig)

print("Images generated.")
