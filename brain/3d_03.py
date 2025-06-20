import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ------------- SETTINGS -------------
n_chains = 3
steps = 200
step_size = 0.1
bounds = (-1, 1)

# ------------- DATA STRUCTURES -------------
# Each chain has a current position
chains = [np.random.uniform(bounds[0], bounds[1], size=3) for _ in range(n_chains)]
# Track all sample positions & colors
all_positions = []  # shape: (frame, chain, 3)
all_colors = []     # shape: (frame, chain)

accepted_inside_count = 0
total_count = 0
pi_estimates = []

# ------------- MCMC STEP FUNCTION -------------
def metropolis_step_3d(current):
    """Propose a random step in 3D, accept or reject."""
    proposal = current + np.random.normal(0, step_size, size=3)
    
    # Out-of-bounds => reject immediately
    if np.any(proposal < bounds[0]) or np.any(proposal > bounds[1]):
        return current, False, False
    
    # We'll do a naive 'always accept' approach for demonstration,
    # so the acceptance check is only about "inside or outside the sphere"
    # but not "reject" in the Metropolis sense of acceptance ratio.
    # (If you want full Metropolis, you'd compute acceptance probability, etc.)
    inside = (proposal[0]**2 + proposal[1]**2 + proposal[2]**2) <= 1.0
    return proposal, True, inside

# ------------- ANIMATION SETUP -------------
fig = plt.figure(figsize=(10,5))

# Left: 3D scatter
ax3d = fig.add_subplot(1, 2, 1, projection='3d')
ax3d.set_title("3D MCMC Sampling")
ax3d.set_xlim(bounds)
ax3d.set_ylim(bounds)
ax3d.set_zlim(bounds)
scat = ax3d.scatter([], [], [], c=[], s=10)

# Right: running π estimate
axpi = fig.add_subplot(1, 2, 2)
axpi.set_title("Running π Estimate")
axpi.set_xlim(0, steps)
axpi.set_ylim(2.5, 4.0)
line_pi, = axpi.plot([], [], lw=2, color='blue')
axpi.set_xlabel("Iteration")
axpi.set_ylabel("π Estimate")

# ------------- UPDATE FUNCTION -------------
def update(frame):
    global accepted_inside_count, total_count
    
    # 1) For each chain, do a step
    frame_positions = []
    frame_colors = []
    
    for i in range(n_chains):
        current_pos = chains[i]
        proposal, accepted, inside = metropolis_step_3d(current_pos)
        
        if accepted:
            chains[i] = proposal  # Move the chain
            # Classify color
            color = 'green' if inside else 'orange'
            # Count if inside
            if inside:
                accepted_inside_count += 1
        else:
            # Rejected
            proposal = current_pos  # remain in place
            color = 'red'
        
        total_count += 1
        
        frame_positions.append(proposal)
        frame_colors.append(color)
    
    all_positions.append(frame_positions)
    all_colors.append(frame_colors)
    
    # 2) Update π estimate
    if total_count > 0:
        pi_est = (accepted_inside_count / total_count) * 6.0  # 6 = 8 / (4/3)
        pi_estimates.append(pi_est)
    else:
        pi_estimates.append(0)
    
    # 3) Update 3D scatter data
    # Flatten all positions up to current frame
    flat_positions = np.array(all_positions).reshape(-1, 3)
    flat_colors = np.array(all_colors).reshape(-1)
    xs, ys, zs = flat_positions[:, 0], flat_positions[:, 1], flat_positions[:, 2]
    
    # Trick for updating scatter in 3D:
    scat._offsets3d = (xs, ys, zs)
    scat.set_color(flat_colors)
    
    # 4) Update π line
    line_pi.set_data(range(len(pi_estimates)), pi_estimates)
    
    # Return the artists to update
    return [scat, line_pi]

# ------------- RUN ANIMATION -------------
ani = FuncAnimation(fig, update, frames=steps, interval=100, blit=False)

plt.tight_layout()
plt.show()
