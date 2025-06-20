import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import matplotlib.animation as animation
import math
import time

# Total number of points for the simulation
total_points = 10000
# Number of frames for the animation (each frame adds a batch of points)
frames = 100
points_per_frame = total_points // frames

# Pre-generate all random points in the cube [-1, 1]^3
points = np.random.uniform(-1, 1, size=(total_points, 3))
# Compute distances from the origin for all points
distances = np.linalg.norm(points, axis=1)
# Boolean array indicating if each point is inside the unit sphere
inside = distances <= 1

# Lists to store the running estimation of π
pi_estimates = []
point_counts = []

# Set up the figure with two subplots: one 3D plot and one line plot
fig = plt.figure(figsize=(14, 6))

# 3D scatter plot for the simulation
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Monte Carlo 3D Simulation")
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_zlim(-1, 1)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
inside_scatter = ax1.scatter([], [], [], color='blue', alpha=0.5, label="Inside Sphere")
outside_scatter = ax1.scatter([], [], [], color='red', alpha=0.5, label="Outside Sphere")
ax1.legend()

# Line plot for π estimation
ax2 = fig.add_subplot(122)
ax2.set_title("Running Estimate of π")
ax2.set_xlim(0, total_points)
ax2.set_ylim(0, 4)
ax2.set_xlabel("Number of Points")
ax2.set_ylabel("Estimated π")
# Horizontal line showing the actual π value
ax2.axhline(math.pi, color='green', linestyle='--', label="Actual π")
line_estimate, = ax2.plot([], [], color='blue', label="Estimated π")
ax2.legend()

# Data containers for scatter plot and line plot
inside_x, inside_y, inside_z = [], [], []
outside_x, outside_y, outside_z = [], [], []
x_line, y_line = [], []

# Record the start time
start_time = time.time()

def init():
    """Initialize the scatter and line plots."""
    inside_scatter._offsets3d = ([], [], [])
    outside_scatter._offsets3d = ([], [], [])
    line_estimate.set_data([], [])
    return inside_scatter, outside_scatter, line_estimate

def update(frame):
    """Update function for the animation."""
    # Determine the range of points to include in this frame
    start_idx = frame * points_per_frame
    end_idx = start_idx + points_per_frame
    if end_idx > total_points:
        end_idx = total_points

    global inside_x, inside_y, inside_z, outside_x, outside_y, outside_z
    
    # Batch points and whether they are inside the sphere
    batch_points = points[start_idx:end_idx]
    batch_inside = inside[start_idx:end_idx]
    
    # Append new points to the respective lists
    for pt, flag in zip(batch_points, batch_inside):
        if flag:
            inside_x.append(pt[0])
            inside_y.append(pt[1])
            inside_z.append(pt[2])
        else:
            outside_x.append(pt[0])
            outside_y.append(pt[1])
            outside_z.append(pt[2])
    
    inside_scatter._offsets3d = (inside_x, inside_y, inside_z)
    outside_scatter._offsets3d = (outside_x, outside_y, outside_z)

    # Update running estimate for π
    current_count = end_idx
    count_inside = np.sum(inside[:end_idx])
    ratio = count_inside / current_count
    # Using ratio = pi/6 => estimated_pi = 6 * ratio
    current_pi_estimate = 6 * ratio
    point_counts.append(current_count)
    pi_estimates.append(current_pi_estimate)
    line_estimate.set_data(point_counts, pi_estimates)
    
    ax2.set_xlim(0, total_points)
    ax2.set_ylim(0, 4)
    
    # If last frame, calculate and print the final results.
    if frame == frames - 1:
        final_pi = current_pi_estimate
        error_percent = abs(final_pi - math.pi) / math.pi * 100
        elapsed_time = time.time() - start_time
        print(f"Final Estimated π: {final_pi:.6f}")
        print(f"Actual π: {math.pi:.6f}")
        print(f"Error Percentage: {error_percent:.4f}%")
        print(f"Time Taken: {elapsed_time:.4f} seconds")
    
    return inside_scatter, outside_scatter, line_estimate

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                              interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()
