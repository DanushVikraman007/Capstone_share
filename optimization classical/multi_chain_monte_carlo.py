import numpy as num
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Parameters
n_of_steps = 10000
step_size = 0.5
burn_in = 1000
n_of_chains = 3
accutal_pi_value = num.pi

def estimate_pi_mcmc_chain(n_of_steps, step_size, burn_in):
    x, y = 0.0, 0.0
    traj_x, traj_y = [], []
    inside_flags = []
    accepted_flags = []
    pi_running_estimate = []
    inside_count = 0

    for i in range(n_of_steps):
        x_new = x + num.random.uniform(-step_size, step_size)
        y_new = y + num.random.uniform(-step_size, step_size)

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
        traj_x.append(x)
        traj_y.append(y)

    return (
        4 * num.sum(inside_flags[burn_in:]) / len(inside_flags[burn_in:]),
        num.array(traj_x),
        num.array(traj_y),
        num.array(inside_flags),
        num.array(accepted_flags),
        num.array(pi_running_estimate)
    )

# Run MCMC Chains
start_time = time.time()
all_trajectories = []
all_running_estimates = []
all_estimates = []
all_inside_flags = []
all_accepted_flags = []

colors = ['red', 'green', 'blue', 'orange', 'purple']

for i in range(n_of_chains):
    pi_est, x_traj, y_traj, inside_flags, accepted_flags, running_pi = estimate_pi_mcmc_chain(n_of_steps, step_size, burn_in)
    all_trajectories.append((x_traj, y_traj))
    all_running_estimates.append(running_pi)
    all_estimates.append(pi_est)
    all_inside_flags.append(inside_flags)
    all_accepted_flags.append(accepted_flags)
    error = abs(pi_est - accutal_pi_value)
    print(f"Chain {i+1} π estimate: {pi_est:.6f} | Error: {error:.6f} (True π: {accutal_pi_value:.6f})")

avg_pi = num.mean(all_estimates)
end_time = time.time()
print(f"\nFinal Average Estimated π over {n_of_chains} chains: {avg_pi:.6f}")
print(f"Total computation time: {end_time - start_time:.2f} seconds")
# Calculate final error of average π
abs_error = abs(avg_pi - accutal_pi_value)
percentage_error = (abs_error / accutal_pi_value) * 100

print(f"Absolute Error of Average π: {abs_error:.6f}")
print(f"Percentage Error of Average π: {percentage_error:.4f}%")


# === Dashboard Layout ===
fig, (ax_path, ax_heatmap, ax_pi) = plt.subplots(1, 3, figsize=(20, 6))

# Sampling Paths
circle = plt.Circle((0, 0), 1, color='lightblue', alpha=0.3, label='Unit Circle')
ax_path.add_patch(circle)
scatters = []
for i in range(n_of_chains):
    scat = ax_path.scatter([], [], s=10, color=colors[i], label=f'Chain {i+1}', alpha=0.7)
    scatters.append(scat)

avg_text = ax_path.text(0, -1.3, '', ha='center', fontsize=11)
ax_path.set_xlim([-1.2, 1.2])
ax_path.set_ylim([-1.5, 1.2])
ax_path.set_aspect('equal')
ax_path.set_title("MCMC Sampling Paths")
ax_path.legend()

# Heatmap (Scatter-based categories)
inside_scatter = ax_heatmap.scatter([], [], color='green', s=6, alpha=0.5, label='Accepted Inside')
outside_scatter = ax_heatmap.scatter([], [], color='orange', s=6, alpha=0.5, label='Accepted Outside')
rejected_scatter = ax_heatmap.scatter([], [], color='gray', s=6, alpha=0.4, label='Rejected')

ax_heatmap.add_patch(plt.Circle((0, 0), 1, color='lightblue', alpha=0.3))
ax_heatmap.set_xlim([-1.2, 1.2])
ax_heatmap.set_ylim([-1.5, 1.2])
ax_heatmap.set_aspect('equal')
ax_heatmap.set_title("Heatmap of Points")
ax_heatmap.legend()

# π Estimate Plot
ax_pi.set_xlim(0, n_of_steps)
ax_pi.set_ylim(accutal_pi_value - 0.5, accutal_pi_value + 0.5)
ax_pi.axhline(accutal_pi_value, color='black', linestyle='--', label='True π')
pi_lines = []
for i in range(n_of_chains):
    line, = ax_pi.plot([], [], color=colors[i], alpha=0.7, label=f'Chain {i+1} π')
    pi_lines.append(line)
avg_pi_line, = ax_pi.plot([], [], color='purple', linewidth=1, label='Avg π')
ax_pi.set_title("Estimated π vs True π")
ax_pi.set_xlabel("Steps")
ax_pi.set_ylabel("π Estimate")
ax_pi.legend()

# Storage
inside_x, inside_y = [], []
outside_x, outside_y = [], []
rejected_x, rejected_y = [], []
x_pi_vals, y_avg_vals = [], []
chain_pi_x = [[] for _ in range(n_of_chains)]
chain_pi_y = [[] for _ in range(n_of_chains)]

# Animation Update
def update(frame):
    for i, scat in enumerate(scatters):
        x_vals = all_trajectories[i][0][:frame]
        y_vals = all_trajectories[i][1][:frame]
        scat.set_offsets(num.column_stack((x_vals, y_vals)))

    for i in range(n_of_chains):
        for j in range(frame-10 if frame >= 10 else 0, frame):
            if j >= len(all_trajectories[i][0]):
                continue
            x, y = all_trajectories[i][0][j], all_trajectories[i][1][j]
            accepted = all_accepted_flags[i][j]
            inside = all_inside_flags[i][j]
            if accepted:
                if inside:
                    inside_x.append(x)
                    inside_y.append(y)
                else:
                    outside_x.append(x)
                    outside_y.append(y)
            else:
                rejected_x.append(x)
                rejected_y.append(y)

    inside_scatter.set_offsets(num.column_stack((inside_x, inside_y)))
    outside_scatter.set_offsets(num.column_stack((outside_x, outside_y)))
    rejected_scatter.set_offsets(num.column_stack((rejected_x, rejected_y)))

    for i in range(n_of_chains):
        if frame < len(all_running_estimates[i]):
            pi_val = all_running_estimates[i][frame - 1]
        else:
            pi_val = all_running_estimates[i][-1]
        chain_pi_x[i].append(frame)
        chain_pi_y[i].append(pi_val)
        pi_lines[i].set_data(chain_pi_x[i], chain_pi_y[i])

    avg_pi_now = num.mean([r[frame - 1] if frame < len(r) else r[-1] for r in all_running_estimates])
    x_pi_vals.append(frame)
    y_avg_vals.append(avg_pi_now)
    avg_pi_line.set_data(x_pi_vals, y_avg_vals)
    avg_text.set_text(f"Step {frame}, Avg π ≈ {avg_pi_now:.6f}")

    return scatters + pi_lines + [inside_scatter, outside_scatter, rejected_scatter, avg_pi_line, avg_text]

# Initialization
def init():
    for scat in scatters:
        scat.set_offsets(num.empty((0, 2)))
    for line in pi_lines:
        line.set_data([], [])
    avg_pi_line.set_data([], [])
    inside_scatter.set_offsets(num.empty((0, 2)))
    outside_scatter.set_offsets(num.empty((0, 2)))
    rejected_scatter.set_offsets(num.empty((0, 2)))
    avg_text.set_text('')
    x_pi_vals.clear()
    y_avg_vals.clear()
    inside_x.clear()
    inside_y.clear()
    outside_x.clear()
    outside_y.clear()
    rejected_x.clear()
    rejected_y.clear()
    for i in range(n_of_chains):
        chain_pi_x[i].clear()
        chain_pi_y[i].clear()
    return scatters + pi_lines + [inside_scatter, outside_scatter, rejected_scatter, avg_pi_line, avg_text]

# Create Animation
an = animation.FuncAnimation(
    fig, update, frames=range(10, n_of_steps, 10),
    init_func=init, blit=True, interval=50, repeat=False
)

plt.tight_layout()
plt.show()
