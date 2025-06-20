import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol, Halton

# Function to estimate Pi
def estimate_pi(samples):
    """Estimate Pi using Monte Carlo method given sample points (Nx2 array)."""
    x, y = samples[:, 0], samples[:, 1]
    inside_circle = np.sum(x**2 + y**2 <= 1)  # Count points inside the unit circle
    return 4 * inside_circle / len(samples)  # Monte Carlo estimate of Pi

# Number of samples
n_samples = 1000
exact_pi = np.pi

# Generate samples
random_samples = np.random.rand(n_samples, 2)

sobol = Sobol(d=2, scramble=True)
sobol_samples = sobol.random(n_samples)

halton = Halton(d=2, scramble=True)
halton_samples = halton.random(n_samples)

# Compute Pi estimates
random_pi_estimate = estimate_pi(random_samples)
sobol_pi_estimate = estimate_pi(sobol_samples)
halton_pi_estimate = estimate_pi(halton_samples)

# Print Results
print(f"Exact Pi: {exact_pi}")
print(f"Random Sampling Pi Estimate: {random_pi_estimate:.6f}")
print(f"Sobol Sampling Pi Estimate: {sobol_pi_estimate:.6f}")
print(f"Halton Sampling Pi Estimate: {halton_pi_estimate:.6f}")

# Plot sample distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Function to plot samples
def plot_samples(ax, samples, title, color):
    ax.scatter(samples[:, 0], samples[:, 1], s=5, c=color, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    circle = plt.Circle((0.5, 0.5), 0.5, color='r', alpha=0.3, fill=True)
    ax.add_patch(circle)

# Plot each sampling method
plot_samples(axes[0], random_samples, "Random Sampling", 'blue')
plot_samples(axes[1], sobol_samples, "Sobol Sampling", 'black')  # Updated Sobol color to black
plot_samples(axes[2], halton_samples, "Halton Sampling", 'green')

plt.show()
