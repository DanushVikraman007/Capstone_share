import numpy as np
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# Parameters
N_spin = 100               # Number of spin_value

J = 1.0               # Interaction strength

Temp_init = 2.0          # Initial temperature

total_num_steps = 300     # Max steps to store in history

Boltz = 1.0             # Boltzmann constant

# Initialize spin_value randomly
def init_spins():

    return np.random.choice([-1, 1], size=N_spin)

# Energy function
def energy(spin_value):

    return -J * np.sum(spin_value * np.roll(spin_value, 1))

# Metropolis single spin flip step
def metropolis_step(spin_value, T):

    i = np.random.randint(0, N_spin)

    s_i = spin_value[i]
    neighbors = spin_value[(i - 1) % N_spin] + spin_value[(i + 1) % N_spin]
    delta_E = 2 * J * s_i * neighbors
    if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / (Boltz * T)):
        spin_value[i] *= -1
    return spin_value

# Initialize simulation
spin_value = init_spins()
energy_value = [energy(spin_value)]
magnetizationVal = [np.mean(spin_value)]
spin_history = [spin_value.copy()]

# Setup figure
fig = pl.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, height_ratios=[1.5, 1, 1, 0.3])

# --- Bar Plot of current spin_value ---
ax_bar = fig.add_subplot(gs[0, :])
bar_rects = ax_bar.bar(np.arange(N_spin), spin_value, color=['red' if s == 1 else 'blue' for s in spin_value])
ax_bar.set_ylim(-1.2, 1.2)
ax_bar.set_title('Real-Time Spin Flip Dynamics (1D Ising)')
ax_bar.set_ylabel('Spin Value')
ax_bar.set_xlabel('Spin Index')

# --- Heatmap of spin history ---
ax_heatmap = fig.add_subplot(gs[1, :])
heatmap = ax_heatmap.imshow(np.array(spin_history).T, aspect='auto', cmap='coolwarm', interpolation='nearest')
ax_heatmap.set_title('Spin Configuration History (Heatmap)')
ax_heatmap.set_ylabel('Spin Index')
ax_heatmap.set_xlabel('MCMC Step')
cbar = pl.colorbar(heatmap, ax=ax_heatmap, orientation='vertical')
cbar.set_label('Spin State (+1 / -1)')

# --- Energy Plot ---
ax_energy = fig.add_subplot(gs[2, 0])
line_energy, = ax_energy.plot([], [], color='orange')
ax_energy.set_title('Energy vs Step')
ax_energy.set_xlim(0, total_num_steps)
ax_energy.set_ylim(-N_spin, N_spin)
ax_energy.set_ylabel('Energy')
ax_energy.set_xlabel('Step')
ax_energy.grid()

# --- Magnetization Plot ---
ax_mag = fig.add_subplot(gs[2, 1])
line_mag, = ax_mag.plot([], [], color='green')
ax_mag.set_title('Magnetization vs Step')
ax_mag.set_xlim(0, total_num_steps)
ax_mag.set_ylim(-1.1, 1.1)
ax_mag.set_ylabel('⟨M⟩')
ax_mag.set_xlabel('Step')
ax_mag.grid()

# --- Slider and Reset Button ---
ax_slider = fig.add_subplot(gs[3, 0])
temp_slider = Slider(ax_slider, 'Temperature (T)', 0.1, 5.0, valinit=Temp_init)

ax_button = fig.add_subplot(gs[3, 1])
reset_button = Button(ax_button, 'Reset Simulation', color='lightgray')

# State variables
T_current = [Temp_init]
step_counter = [0]

# -------- Animation Update --------
def update(frame):
    global spin_value
    T = T_current[0]
    spin_value = metropolis_step(spin_value, T)

    # Update values
    E = energy(spin_value)
    M = np.mean(spin_value)
    energy_value.append(E)
    magnetizationVal.append(M)
    spin_history.append(spin_value.copy())

    # Keep history size under control
    if len(spin_history) > total_num_steps:
        del spin_history[0]
        del energy_value[0]
        del magnetizationVal[0]

    # Update bar plot
    for rect, s in zip(bar_rects, spin_value):
        rect.set_height(s)
        rect.set_color('red' if s == 1 else 'blue')

    # Update heatmap
    heatmap.set_data(np.array(spin_history).T)
    heatmap.set_extent([0, len(spin_history), 0, N_spin])

    # Update energy/magnetization plots
    line_energy.set_data(np.arange(len(energy_value)), energy_value)
    ax_energy.set_xlim(0, len(energy_value))
    ax_energy.set_ylim(min(energy_value) - 5, max(energy_value) + 5)

    line_mag.set_data(np.arange(len(magnetizationVal)), magnetizationVal)
    ax_mag.set_xlim(0, len(magnetizationVal))
    ax_mag.set_ylim(-1.1, 1.1)

    step_counter[0] += 1
    return bar_rects

# -------- Slider and Button Handlers --------
def slider_changed(val):
    T_current[0] = temp_slider.val

def reset(event):
    global spin_value, energy_value, magnetizationVal, spin_history
    spin_value = init_spins()
    energy_value.clear()
    magnetizationVal.clear()
    spin_history.clear()
    energy_value.append(energy(spin_value))
    magnetizationVal.append(np.mean(spin_value))
    spin_history.append(spin_value.copy())

    for rect, s in zip(bar_rects, spin_value):
        rect.set_height(s)
        rect.set_color('red' if s == 1 else 'blue')

    line_energy.set_data([], [])
    line_mag.set_data([], [])
    heatmap.set_data(np.array(spin_history).T)

    fig.canvas.draw_idle()

temp_slider.on_changed(slider_changed)
reset_button.on_clicked(reset)

# Run Animation
animate = animation.FuncAnimation(fig, update, interval=50, blit=False)
pl.tight_layout()
pl.show()
