import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import time
import random

class QuantumMCMC_PiEstimator:
    def __init__(self, n_qubits=4, n_iterations=1000, n_chains=5):
        self.n_qubits = n_qubits
        self.n_iterations = n_iterations
        self.n_chains = n_chains
        # Updated for newer Qiskit versions
        self.backend = AerSimulator()
        self.chain_histories = []
        self.pi_estimates = []
        
    def create_quantum_random_circuit(self):
        """Create a quantum circuit for generating random numbers"""
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
            
        return qc
    
    def quantum_random_point(self):
        """Generate a random point using quantum circuit"""
        qc = self.create_quantum_random_circuit()
        
        # Transpile and run the circuit
        transpiled_qc = transpile(qc, self.backend)
        job = self.backend.run(transpiled_qc, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Get the most frequent measurement result (should be only one with shots=1)
        binary_string = max(counts, key=counts.get)
        
        # Handle case where we have odd number of qubits
        mid_point = self.n_qubits // 2
        x_bits = binary_string[:mid_point] if mid_point > 0 else '0'
        y_bits = binary_string[mid_point:] if len(binary_string) > mid_point else '0'
        
        # Handle edge case for single bit
        if len(x_bits) == 0:
            x_bits = '0'
        if len(y_bits) == 0:
            y_bits = '0'
            
        # Convert to coordinates in [-1, 1] x [-1, 1]
        x_max = 2**len(x_bits) - 1
        y_max = 2**len(y_bits) - 1
        
        # Avoid division by zero
        if x_max == 0:
            x = 0
        else:
            x = (int(x_bits, 2) / x_max) * 2 - 1
            
        if y_max == 0:
            y = 0
        else:
            y = (int(y_bits, 2) / y_max) * 2 - 1
        
        return x, y
    
    def mcmc_step(self, current_x, current_y, step_size=0.1):
        """Perform one MCMC step with quantum randomness"""
        # Generate quantum random proposal
        qx, qy = self.quantum_random_point()
        
        # Propose new state with smaller step size
        new_x = current_x + step_size * qx
        new_y = current_y + step_size * qy
        
        # Keep points within the square [-1, 1] x [-1, 1]
        new_x = np.clip(new_x, -1, 1)
        new_y = np.clip(new_y, -1, 1)
        
        # Simple acceptance (always accept for this demonstration)
        # In a more sophisticated version, you could implement Metropolis-Hastings
        return new_x, new_y
    
    def run_single_chain(self, chain_id):
        """Run a single Markov chain"""
        print(f"Running chain {chain_id + 1}/{self.n_chains}...")
        
        # Initialize random starting point
        x, y = self.quantum_random_point()
        
        chain_x = [x]
        chain_y = [y]
        inside_circle = []
        
        points_inside = 0
        
        for i in range(self.n_iterations):
            # MCMC step
            x, y = self.mcmc_step(x, y)
            chain_x.append(x)
            chain_y.append(y)
            
            # Check if point is inside unit circle
            if x**2 + y**2 <= 1:
                points_inside += 1
                inside_circle.append(True)
            else:
                inside_circle.append(False)
            
            # Estimate π periodically
            if (i + 1) % 50 == 0:
                pi_estimate = 4 * points_inside / (i + 1)
                self.pi_estimates.append(pi_estimate)
        
        # Store chain history
        self.chain_histories.append({
            'x': chain_x,
            'y': chain_y,
            'inside': inside_circle,
            'final_pi': 4 * points_inside / self.n_iterations
        })
        
        return 4 * points_inside / self.n_iterations
    
    def run_estimation(self):
        """Run the complete π estimation with multiple chains"""
        print("Starting Quantum Monte Carlo Markov Chain π Estimation...")
        print(f"Using {self.n_qubits} qubits, {self.n_iterations} iterations, {self.n_chains} chains")
        
        start_time = time.time()
        
        pi_estimates_per_chain = []
        
        for chain_id in range(self.n_chains):
            pi_est = self.run_single_chain(chain_id)
            pi_estimates_per_chain.append(pi_est)
        
        end_time = time.time()
        
        # Calculate final estimate as average of all chains
        final_pi_estimate = np.mean(pi_estimates_per_chain)
        
        # Calculate statistics
        actual_pi = np.pi
        error = abs(final_pi_estimate - actual_pi)
        error_percentage = (error / actual_pi) * 100
        execution_time = end_time - start_time
        
        # Print results
        print("\n" + "="*50)
        print("QUANTUM MONTE CARLO MARKOV CHAIN RESULTS")
        print("="*50)
        print(f"Actual π:           {actual_pi:.6f}")
        print(f"Estimated π:        {final_pi_estimate:.6f}")
        print(f"Error:              {error:.6f}")
        print(f"Error percentage:   {error_percentage:.4f}%")
        print(f"Execution time:     {execution_time:.2f} seconds")
        print(f"Per chain estimates: {[f'{est:.4f}' for est in pi_estimates_per_chain]}")
        
        return final_pi_estimate, error, error_percentage, execution_time
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Markov Chain Paths
        plt.subplot(2, 3, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_chains))
        
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        plt.plot(circle_x, circle_y, 'k-', linewidth=2, label='Unit Circle')
        
        # Plot chain paths
        for i, (chain, color) in enumerate(zip(self.chain_histories, colors)):
            plt.plot(chain['x'][::20], chain['y'][::20], 'o-', 
                    color=color, alpha=0.7, markersize=2, linewidth=0.5,
                    label=f'Chain {i+1}')
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Markov Chain Paths (Every 20th Point)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 2. Point Distribution
        plt.subplot(2, 3, 2)
        all_x = []
        all_y = []
        all_inside = []
        
        for chain in self.chain_histories:
            all_x.extend(chain['x'][1:])  # Skip initial point
            all_y.extend(chain['y'][1:])
            all_inside.extend(chain['inside'])
        
        # Plot points inside and outside circle
        inside_x = [x for x, inside in zip(all_x, all_inside) if inside]
        inside_y = [y for y, inside in zip(all_y, all_inside) if inside]
        outside_x = [x for x, inside in zip(all_x, all_inside) if not inside]
        outside_y = [y for y, inside in zip(all_y, all_inside) if not inside]
        
        plt.scatter(inside_x, inside_y, c='red', s=1, alpha=0.6, label='Inside Circle')
        plt.scatter(outside_x, outside_y, c='blue', s=1, alpha=0.6, label='Outside Circle')
        plt.plot(circle_x, circle_y, 'k-', linewidth=2)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('All MCMC Sample Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 3. π Convergence
        plt.subplot(2, 3, 3)
        if self.pi_estimates:
            plt.plot(range(50, len(self.pi_estimates)*50 + 1, 50), 
                    self.pi_estimates, 'b-', linewidth=2, label='π Estimate')
            plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label='True π')
            plt.xlabel('Iterations')
            plt.ylabel('π Estimate')
            plt.title('π Estimation Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Chain Statistics
        plt.subplot(2, 3, 4)
        chain_pi_estimates = [chain['final_pi'] for chain in self.chain_histories]
        plt.bar(range(1, self.n_chains + 1), chain_pi_estimates, 
                color=colors, alpha=0.7)
        plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label='True π')
        plt.xlabel('Chain Number')
        plt.ylabel('π Estimate')
        plt.title('π Estimates per Chain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Error Analysis
        plt.subplot(2, 3, 5)
        errors = [abs(est - np.pi) for est in chain_pi_estimates]
        plt.bar(range(1, self.n_chains + 1), errors, color='orange', alpha=0.7)
        plt.xlabel('Chain Number')
        plt.ylabel('Absolute Error')
        plt.title('Estimation Error per Chain')
        plt.grid(True, alpha=0.3)
        
        # 6. Quantum Circuit Visualization
        plt.subplot(2, 3, 6)
        qc = self.create_quantum_random_circuit()
        
        # Create a simple text representation
        plt.text(0.1, 0.8, "Quantum Random Number Generator Circuit:", 
                fontsize=12, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f"• {self.n_qubits} qubits with Hadamard gates", 
                fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f"• CNOT gates for entanglement", 
                fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f"• Measurement of all qubits", 
                fontsize=10, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f"• Converts to 2D coordinates", 
                fontsize=10, transform=plt.gca().transAxes)
        
        # Add some quantum circuit elements visually
        plt.plot([0.1, 0.9], [0.25, 0.25], 'k-', linewidth=2)
        plt.plot([0.1, 0.9], [0.15, 0.15], 'k-', linewidth=2)
        plt.text(0.2, 0.27, 'H', fontsize=12, ha='center')
        plt.text(0.2, 0.17, 'H', fontsize=12, ha='center')
        plt.plot([0.4, 0.4], [0.13, 0.27], 'k-', linewidth=2)
        plt.plot(0.4, 0.25, 'ko', markersize=8)
        plt.plot(0.4, 0.15, 'k+', markersize=10)
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Quantum Circuit Architecture')
        
        plt.tight_layout()
        plt.show()

# Run the estimation
if __name__ == "__main__":
    # Initialize estimator
    estimator = QuantumMCMC_PiEstimator(
        n_qubits=4,          # 4 qubits for random number generation
        n_iterations=500,    # 500 iterations per chain
        n_chains=3           # 3 independent chains
    )
    
    # Run estimation
    pi_est, error, error_pct, exec_time = estimator.run_estimation()
    
    # Create visualizations
    estimator.visualize_results()