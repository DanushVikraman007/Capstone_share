import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import time

# Parameters
n_steps = 10000
step_size = 0.5
burn_in = 1000

# === MCMC π Estimation Function ===
def estimate_pi_mcmc_chain(n_steps, step_size, burn_in):
    x, y = 0.0, 0.0
    trajectory_x, trajectory_y = [], []
    inside_flags = []
    accepted_flags = []
    pi_running_estimate = []

    inside_count = 0
    for i in range(n_steps):
        x_new = x + np.random.uniform(-step_size, step_size)
        y_new = y + np.random.uniform(-step_size, step_size)

        if -1 <= x_new <= 1 and -1 <= y_new <= 1:
            x, y = x_new, y_new
            accepted_flags.append(True)
        else:
            accepted_flags.append(False)

        is_inside = x**2 + y**2 <= 1
        inside_flags.append(is_inside)

        if is_inside:
            inside_count += 1

        pi_running_estimate.append(4 * inside_count / (i + 1))
        trajectory_x.append(x)
        trajectory_y.append(y)

    inside_flags = np.array(inside_flags)
    accepted_flags = np.array(accepted_flags)
    pi_running_estimate = np.array(pi_running_estimate)

    pi_est_post_burn = 4 * np.sum(inside_flags[burn_in:]) / len(inside_flags[burn_in:])

    return pi_est_post_burn, np.array(trajectory_x), np.array(trajectory_y), inside_flags, accepted_flags, pi_running_estimate

# === Run Chain ===
start_time = time.time()
pi_est, traj_x, traj_y, inside_flags, accepted_flags, running_pi = estimate_pi_mcmc_chain(n_steps, step_size, burn_in)
end_time = time.time()

# Calculate Error and Time
pi_error = abs(pi_est - np.pi)
total_time = end_time - start_time

# Print results
print(f"\nFinal Estimated π (post burn-in): {pi_est:.6f}")
print(f"π Estimation Error: {pi_error:.6f}")
print(f"Time Taken: {total_time:.2f} seconds")
percentage_error = (pi_error / np.pi) * 100
print(f"percentage error: {percentage_error:.2f}% ")
# === Setup Plots ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Left: Walkthrough inside unit circle
circle = plt.Circle((0, 0), 1, color='lightblue', alpha=0.3)
ax1.add_patch(circle)
ax1.set_xlim([-1.1, 1.1])
ax1.set_ylim([-1.1, 1.1])
ax1.set_title("MCMC Walkthrough (Single Chain)")
ax1.set_aspect('equal')

# Path line segments (gradient)
points = np.array([traj_x, traj_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(0, len(segments))
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(np.arange(len(segments)))
lc.set_linewidth(0.5)
line = ax1.add_collection(lc)

# Scattered classified points
inside_scatter = ax1.scatter([], [], color='blue', label='Accepted Inside', s=15)
outside_scatter = ax1.scatter([], [], color='orange', label='Accepted Outside', s=15)
rejected_scatter = ax1.scatter([], [], color='red', label='Rejected', s=15)
ax1.legend()

# Right: π estimation over time
est_line, = ax2.plot([], [], label='Estimated π', color='black')
actual_pi = ax2.axhline(np.pi, color='red', linestyle='--', label='Actual π')
ax2.set_xlim(0, n_steps)
ax2.set_ylim(2.5, 4.5)
ax2.set_xlabel("Steps")
ax2.set_ylabel("π Estimate")
ax2.set_title("Estimated π vs Steps")
ax2.legend()
ax2.grid(True)

# === Animation Functions ===
def init():
    est_line.set_data([], [])
    lc.set_segments([])
    inside_scatter.set_offsets(np.empty((0, 2)))
    outside_scatter.set_offsets(np.empty((0, 2)))
    rejected_scatter.set_offsets(np.empty((0, 2)))
    return est_line, lc, inside_scatter, outside_scatter, rejected_scatter

def update(frame):
    # Trail line
    if frame > 1:
        segs = np.concatenate([
            np.array([traj_x[:frame], traj_y[:frame]]).T[:-1, None],
            np.array([traj_x[:frame], traj_y[:frame]]).T[1:, None]
        ], axis=1)
        lc.set_segments(segs)
        lc.set_array(np.arange(len(segs)))

    # Point classifications
    inside_pts = np.column_stack((traj_x[:frame], traj_y[:frame]))[accepted_flags[:frame] & inside_flags[:frame]]
    outside_pts = np.column_stack((traj_x[:frame], traj_y[:frame]))[accepted_flags[:frame] & ~inside_flags[:frame]]
    rejected_pts = np.column_stack((traj_x[:frame], traj_y[:frame]))[~accepted_flags[:frame]]

    inside_scatter.set_offsets(inside_pts)
    outside_scatter.set_offsets(outside_pts)
    rejected_scatter.set_offsets(rejected_pts)

    # π estimate plot
    est_line.set_data(np.arange(frame), running_pi[:frame])

    return est_line, lc, inside_scatter, outside_scatter, rejected_scatter

ani = animation.FuncAnimation(
    fig, update, frames=range(10, n_steps, 10), init_func=init,
    blit=True, interval=50, repeat=False)

plt.tight_layout()
plt.show()
