import numpy as nm
import matplotlib.pyplot as pt
n=10000
def monte_pi(n):

    X = nm.random.uniform(-1, 1, n)
    
    Y = nm.random.uniform(-1, 1, n)

    
    dist = X**2 + Y**2
    
    inside_points = dist <= 1

    pi_estimates = nm.cumsum(inside_points) / nm.arange(1, n + 1) * 4

  
    pi_final = pi_estimates[-1]

    pt.figure(figsize=(16, 8))

   
    pt.subplot(1, 2, 1)
    pt.scatter(X[inside_points], Y[inside_points], color='green', s=1, label='Inside Circle')
    pt.scatter(X[~inside_points], Y[~inside_points], color='red', s=1, label='Outside Circle')

    
    circle = pt.Circle((0, 0), 1, color='blue', fill=False, linewidth=1.5, label='Unit Circle')
    pt.gca().add_artist(circle)

    pt.title("Monte Carlo Estimation of Pi: Points Visualization")
    pt.xlabel("X")
    pt.ylabel("Y")
    pt.axis("equal")
    pt.grid(alpha=0.3)
    pt.legend()

   
    pt.subplot(1, 2, 2)
    pt.plot(nm.arange(1, n + 1), pi_estimates, color='blue', label='Estimated Pi')
    pt.axhline(y=nm.pi, color='red', linestyle='--', label='Actual Pi')
    pt.title("Convergence of Pi Estimate")
    pt.xlabel("Number of Iterations")
    pt.ylabel("Estimated Pi Value")
    pt.grid(alpha=0.3)
    pt.legend()

    pt.tight_layout()
    pt.show()

   
    return pi_final

pi_final = monte_pi(n)


print(f"Estimated value of Pi: {pi_final:.5f}")