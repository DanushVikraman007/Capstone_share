import numpy as np

import matplotlib.pyplot as pl

from scipy.stats.qmc import Sobol, Halton

def is_inside_circle(X, Y):
    return X**2 + Y**2 <= 1

def monte_carlo_pi(sampler, n_samples):

    samples = sampler.random(n_samples)
    
    X, Y = samples[:, 0], samples[:, 1]
    
    inside_circle = np.sum(is_inside_circle(X, Y))
    
    pi_estimate = 4 * (inside_circle / n_samples)
    
    return pi_estimate

n_samples = 1000

exact_pi = np.pi

random_samples = np.random.rand(n_samples, 2)

random_inside = np.sum(is_inside_circle(random_samples[:, 0], random_samples[:, 1]))

random_pi = 4 * (random_inside / n_samples)


sobol = Sobol(d=2, scramble=True)

sobol_pi = monte_carlo_pi(sobol, n_samples)

halton = Halton(d=2, scramble=True)

halton_pi = monte_carlo_pi(halton, n_samples)

print(f"Exact π: {exact_pi:.6f}")
print(f"Random Sampling Estimate: {random_pi:.6f}")


print(f"Sobol Sampling Estimate: {sobol_pi:.6f}")
print(f"Halton Sampling Estimate: {halton_pi:.6f}")

fig, axes = pl.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(random_samples[:, 0], random_samples[:, 1], s=5, c=['blue' if is_inside_circle(X, Y) else 'red' for X, Y in random_samples])
axes[0].set_title("Random Sampling")

sobol_samples = sobol.random(n_samples)
axes[1].scatter(sobol_samples[:, 0], sobol_samples[:, 1], s=5, c=['blue' if is_inside_circle(X, Y) else 'red' for X, Y in sobol_samples])
axes[1].set_title("Sobol Sampling")

halton_samples = halton.random(n_samples)

axes[2].scatter(halton_samples[:, 0], halton_samples[:, 1], s=5, c=['blue' if is_inside_circle(X, Y) else 'red' for X, Y in halton_samples])

axes[2].set_title("Halton Sampling")


pl.show()
