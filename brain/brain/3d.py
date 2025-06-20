import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting
import matplotlib.animation as animation
import math
import time

# Start timing the simulation
start_time = time.time()

# Total number of points for the simulation
total_points = 5000
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

def init():
    """Initialize the scatter and line plots."""
    inside_scatter._offsets3d = ([], [], [])
    outside_scatter._offsets3d = ([], [], [])
    line_estimate.set_data([], [])
    return inside_scatter, outside_scatter, line_estimate

def update(frame):
    """Update function for the animation."""
    # Determine the range of points to include in this frame
    start = frame * points_per_frame
    end = start + points_per_frame
    if end > total_points:
        end = total_points

    # Update scatter data with new batch of points
    global inside_x, inside_y, inside_z, outside_x, outside_y, outside_z
    batch_points = points[start:end]
    batch_inside = inside[start:end]
    
    # Append new points to the respective lists
    for pt, inside_flag in zip(batch_points, batch_inside):
        if inside_flag:
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
    current_count = end
    count_inside = np.sum(inside[:end])
    ratio = count_inside / current_count
    # Using ratio = pi/6 => estimated_pi = 6 * ratio
    current_pi_estimate = 6 * ratio
    point_counts.append(current_count)
    pi_estimates.append(current_pi_estimate)
    line_estimate.set_data(point_counts, pi_estimates)
    
    ax2.set_xlim(0, total_points)
    ax2.set_ylim(0, 4)
    
    return inside_scatter, outside_scatter, line_estimate

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                              interval=100, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# Calculate final π estimate and error after animation
final_pi = pi_estimates[-1]
error = abs(final_pi - math.pi)
percentage_error = (error / math.pi) * 100
time_taken = time.time() - start_time

# Print results
print(f"Final Estimated π (post burn-in): {final_pi:.6f}")
print(f"π Estimation Error: {error:.6f}")
print(f"Time Taken: {time_taken:.2f} seconds")
print(f"Percentage Error: {percentage_error:.2f}%")
