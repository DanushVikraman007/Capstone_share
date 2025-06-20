import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time
import random
from collections import deque

# Try to import Qiskit components with fallbacks
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
    print("Qiskit imported successfully!")
except ImportError as e:
    print(f"Qiskit import failed: {e}")
    QISKIT_AVAILABLE = False
    print("Running in classical simulation mode...")

class AnimatedQuantumMCMC_PiEstimator:
    def __init__(self, n_qubits=4, n_iterations=1000, n_chains=3, animation_speed=50):
        self.n_qubits = n_qubits
        self.n_iterations = n_iterations
        self.n_chains = n_chains
        self.animation_speed = animation_speed  # milliseconds between frames
        
        self.quantum_mode = QISKIT_AVAILABLE
        if self.quantum_mode:
            try:
                self.backend = AerSimulator()
                print("Using quantum random number generation")
            except Exception as e:
                print(f"Quantum backend failed: {e}")
                self.quantum_mode = False
        
        if not self.quantum_mode:
            print("Using classical random number generation")
        
        # Animation data storage
        self.chains_data = []
        self.pi_history = []
        self.points_inside_history = []
        self.total_points_history = []
        self.current_iteration = 0
        
        # Initialize chains
        for i in range(self.n_chains):
            x, y = self.quantum_random_point()
            self.chains_data.append({
                'x': deque([x], maxlen=100),  # Keep last 100 points for trail
                'y': deque([y], maxlen=100),
                'current_x': x,
                'current_y': y,
                'points_inside': 0,
                'total_points': 0,
                'color': plt.cm.tab10(i)
            })
        
        # Set up the figure and subplots
        self.setup_plots()
    
    def quantum_random_point(self):
        """Generate a random point using quantum circuit or classical fallback"""
        if self.quantum_mode:
            try:
                qc = QuantumCircuit(self.n_qubits, self.n_qubits)
                
                # Apply Hadamard gates for superposition
                for i in range(self.n_qubits):
                    qc.h(i)
                
                # Add some entanglement for better randomness
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                
                # Measure all qubits
                for i in range(self.n_qubits):
                    qc.measure(i, i)
                
                # Execute the quantum circuit
                job = self.backend.run(qc, shots=1)
                result = job.result()
                counts = result.get_counts(qc)
                
                # Get the measurement result
                binary_string = list(counts.keys())[0]
                
            except Exception as e:
                print(f"Quantum execution failed: {e}, using classical fallback")
                binary_string = ''.join([str(random.randint(0,1)) for _ in range(self.n_qubits)])
        else:
            # Classical random generation
            binary_string = ''.join([str(random.randint(0,1)) for _ in range(self.n_qubits)])
        
        # Convert to coordinates in [-1, 1] x [-1, 1]
        half_bits = self.n_qubits // 2
        x_bits = binary_string[:half_bits]
        y_bits = binary_string[half_bits:half_bits*2]
        
        # Handle edge cases
        if len(x_bits) == 0:
            x_bits = '0'
        if len(y_bits) == 0:
            y_bits = '0'
        
        max_val = max(1, 2**len(x_bits) - 1)
        x = (int(x_bits, 2) / max_val) * 2 - 1
        
        max_val = max(1, 2**len(y_bits) - 1)
        y = (int(y_bits, 2) / max_val) * 2 - 1
        
        return x, y
    
    def mcmc_step(self, current_x, current_y, step_size=0.15):
        """Perform one MCMC step"""
        qx, qy = self.quantum_random_point()
        
        # Propose new state
        new_x = current_x + step_size * qx
        new_y = current_y + step_size * qy
        
        # Keep points within bounds
        new_x = np.clip(new_x, -1, 1)
        new_y = np.clip(new_y, -1, 1)
        
        return new_x, new_y
    
    def setup_plots(self):
        """Set up the animated plots"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(f'Animated {"Quantum" if self.quantum_mode else "Classical"} MCMC π Estimation', 
                         fontsize=16, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main MCMC visualization
        self.ax_main = self.fig.add_subplot(gs[0, :2])
        self.ax_main.set_xlim(-1.2, 1.2)
        self.ax_main.set_ylim(-1.2, 1.2)
        self.ax_main.set_xlabel('X')
        self.ax_main.set_ylabel('Y')
        self.ax_main.set_title('MCMC Chain Evolution (Real-time)')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.set_aspect('equal')
        
        # Draw unit circle
        circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        self.ax_main.add_patch(circle)
        
        # Initialize chain plots
        self.chain_lines = []
        self.current_points = []
        for i in range(self.n_chains):
            color = self.chains_data[i]['color']
            # Trail line
            line, = self.ax_main.plot([], [], 'o-', color=color, alpha=0.6, 
                                     markersize=3, linewidth=1, label=f'Chain {i+1}')
            self.chain_lines.append(line)
            
            # Current position marker
            point, = self.ax_main.plot([], [], 'o', color=color, markersize=8, 
                                      markeredgecolor='white', markeredgewidth=1)
            self.current_points.append(point)
        
        self.ax_main.legend(loc='upper right')
        
        # 2. π Convergence plot
        self.ax_pi = self.fig.add_subplot(gs[0, 2])
        self.ax_pi.set_xlabel('Iteration')
        self.ax_pi.set_ylabel('π Estimate')
        self.ax_pi.set_title('π Convergence')
        self.ax_pi.grid(True, alpha=0.3)
        self.ax_pi.axhline(y=np.pi, color='red', linestyle='--', alpha=0.7, label='True π')
        
        self.pi_line, = self.ax_pi.plot([], [], 'b-', linewidth=2, label='π Estimate')
        self.ax_pi.legend()
        
        # 3. Statistics display
        self.ax_stats = self.fig.add_subplot(gs[1, 0])
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.9, '', transform=self.ax_stats.transAxes,
                                           fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 4. Points distribution
        self.ax_dist = self.fig.add_subplot(gs[1, 1])
        self.ax_dist.set_xlabel('Inside Circle')
        self.ax_dist.set_ylabel('Count')
        self.ax_dist.set_title('Points Distribution')
        self.ax_dist.grid(True, alpha=0.3)
        
        # 5. Individual chain π estimates
        self.ax_chains = self.fig.add_subplot(gs[1, 2])
        self.ax_chains.set_xlabel('Chain')
        self.ax_chains.set_ylabel('π Estimate')
        self.ax_chains.set_title('Per Chain π Estimates')
        self.ax_chains.grid(True, alpha=0.3)
        self.ax_chains.axhline(y=np.pi, color='red', linestyle='--', alpha=0.7)
    
    def update_animation(self, frame):
        """Update function for animation"""
        if self.current_iteration >= self.n_iterations:
            return self.chain_lines + self.current_points + [self.pi_line]
        
        # Update each chain
        total_inside = 0
        total_points = 0
        
        for i, chain in enumerate(self.chains_data):
            # MCMC step
            new_x, new_y = self.mcmc_step(chain['current_x'], chain['current_y'])
            
            # Update chain data
            chain['x'].append(new_x)
            chain['y'].append(new_y)
            chain['current_x'] = new_x
            chain['current_y'] = new_y
            chain['total_points'] += 1
            
            # Check if inside circle
            if new_x**2 + new_y**2 <= 1:
                chain['points_inside'] += 1
            
            # Update plot data
            self.chain_lines[i].set_data(list(chain['x']), list(chain['y']))
            self.current_points[i].set_data([new_x], [new_y])
            
            total_inside += chain['points_inside']
            total_points += chain['total_points']
        
        # Calculate current π estimate
        if total_points > 0:
            current_pi = 4 * total_inside / total_points
            self.pi_history.append(current_pi)
            
            # Update π convergence plot
            if len(self.pi_history) > 1:
                self.pi_line.set_data(range(len(self.pi_history)), self.pi_history)
                self.ax_pi.relim()
                self.ax_pi.autoscale_view()
        
        # Update statistics text
        if total_points > 0:
            error = abs(current_pi - np.pi)
            error_pct = (error / np.pi) * 100
            
            stats_str = f"""Current Statistics:
Iteration: {self.current_iteration + 1}/{self.n_iterations}
Mode: {"Quantum" if self.quantum_mode else "Classical"}

Current π estimate: {current_pi:.6f}
True π: {np.pi:.6f}
Error: {error:.6f}
Error %: {error_pct:.3f}%

Total points: {total_points:,}
Points inside: {total_inside:,}
Points outside: {total_points - total_inside:,}

Chains: {self.n_chains}
Qubits/bits: {self.n_qubits}
"""
            self.stats_text.set_text(stats_str)
        
        # Update points distribution
        if self.current_iteration % 20 == 0:  # Update every 20 iterations
            self.ax_dist.clear()
            inside_counts = [chain['points_inside'] for chain in self.chains_data]
            outside_counts = [chain['total_points'] - chain['points_inside'] for chain in self.chains_data]
            
            x = range(1, self.n_chains + 1)
            width = 0.35
            
            self.ax_dist.bar([i - width/2 for i in x], inside_counts, width, 
                           label='Inside', color='red', alpha=0.7)
            self.ax_dist.bar([i + width/2 for i in x], outside_counts, width, 
                           label='Outside', color='blue', alpha=0.7)
            
            self.ax_dist.set_xlabel('Chain')
            self.ax_dist.set_ylabel('Count')
            self.ax_dist.set_title('Points Distribution')
            self.ax_dist.legend()
            self.ax_dist.grid(True, alpha=0.3)
        
        # Update individual chain π estimates
        if self.current_iteration % 30 == 0:  # Update every 30 iterations
            self.ax_chains.clear()
            chain_pi_estimates = []
            for chain in self.chains_data:
                if chain['total_points'] > 0:
                    pi_est = 4 * chain['points_inside'] / chain['total_points']
                    chain_pi_estimates.append(pi_est)
                else:
                    chain_pi_estimates.append(0)
            
            colors = [chain['color'] for chain in self.chains_data]
            bars = self.ax_chains.bar(range(1, self.n_chains + 1), chain_pi_estimates, 
                                    color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, val in zip(bars, chain_pi_estimates):
                height = bar.get_height()
                self.ax_chains.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            self.ax_chains.axhline(y=np.pi, color='red', linestyle='--', alpha=0.7, label='True π')
            self.ax_chains.set_xlabel('Chain')
            self.ax_chains.set_ylabel('π Estimate')
            self.ax_chains.set_title('Per Chain π Estimates')
            self.ax_chains.grid(True, alpha=0.3)
            self.ax_chains.legend()
        
        self.current_iteration += 1
        
        return self.chain_lines + self.current_points + [self.pi_line]
    
    def run_animation(self):
        """Start the animation"""
        print(f"Starting animated {'Quantum' if self.quantum_mode else 'Classical'} MCMC π estimation...")
        print(f"Using {self.n_qubits} {'qubits' if self.quantum_mode else 'bits'}, "
              f"{self.n_iterations} iterations, {self.n_chains} chains")
        print("Close the window to stop the animation.")
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_animation, frames=self.n_iterations,
            interval=self.animation_speed, blit=False, repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        # Print final results
        if self.pi_history:
            final_pi = self.pi_history[-1]
            error = abs(final_pi - np.pi)
            error_pct = (error / np.pi) * 100
            
            print("\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)
            print(f"Final π estimate: {final_pi:.8f}")
            print(f"True π:          {np.pi:.8f}")
            print(f"Error:           {error:.8f}")
            print(f"Error %:         {error_pct:.4f}%")
    
    def save_animation(self, filename='quantum_mcmc_pi.gif', fps=10):
        """Save the animation as GIF"""
        print(f"Saving animation to {filename}...")
        
        # Create a new animation for saving
        save_ani = animation.FuncAnimation(
            self.fig, self.update_animation, frames=min(200, self.n_iterations),
            interval=100, blit=False, repeat=False
        )
        
        # Save as GIF
        save_ani.save(filename, writer='pillow', fps=fps)
        print(f"Animation saved as {filename}")

# Interactive launcher
def run_interactive_demo():
    """Run an interactive demo with user choices"""
    print("="*60)
    print("ANIMATED QUANTUM MCMC π ESTIMATOR")
    print("="*60)
    
    # Get user preferences
    try:
        n_qubits = int(input("Number of qubits/bits (2-6, default 4): ") or "4")
        n_qubits = max(2, min(6, n_qubits))
        
        n_chains = int(input("Number of chains (1-5, default 3): ") or "3")
        n_chains = max(1, min(5, n_chains))
        
        n_iterations = int(input("Iterations per chain (100-2000, default 500): ") or "500")
        n_iterations = max(100, min(2000, n_iterations))
        
        speed = int(input("Animation speed ms (10-200, default 50): ") or "50")
        speed = max(10, min(200, speed))
        
    except ValueError:
        print("Using default values...")
        n_qubits, n_chains, n_iterations, speed = 4, 3, 500, 50
    
    # Create and run estimator
    estimator = AnimatedQuantumMCMC_PiEstimator(
        n_qubits=n_qubits,
        n_iterations=n_iterations,
        n_chains=n_chains,
        animation_speed=speed
    )
    
    estimator.run_animation()
    
    # Option to save
    save_choice = input("\nSave animation as GIF? (y/n): ").lower().strip()
    if save_choice == 'y':
        filename = input("Filename (default: quantum_mcmc_pi.gif): ").strip() or "quantum_mcmc_pi.gif"
        if not filename.endswith('.gif'):
            filename += '.gif'
        estimator.save_animation(filename)

# Run the demo
if __name__ == "__main__":
    run_interactive_demo()