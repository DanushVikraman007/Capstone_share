import numpy as np
import matplotlib.pyplot as plt
import time
import random

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

class QuantumMCMC_PiEstimator:
    def __init__(self, n_qubits=4, n_iterations=1000, n_chains=5):
        self.n_qubits = n_qubits
        self.n_iterations = n_iterations
        self.n_chains = n_chains
        self.chain_histories = []
        self.pi_estimates = []
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
    
    def create_quantum_random_circuit(self):
        """Create a quantum circuit for generating random numbers"""
        if not self.quantum_mode:
            return None
            
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
        """Generate a random point using quantum circuit or classical fallback"""
        if self.quantum_mode:
            try:
                qc = self.create_quantum_random_circuit()
                
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
        
        # Handle case where we might have odd number of qubits
        if len(x_bits) == 0:
            x_bits = '0'
        if len(y_bits) == 0:
            y_bits = '0'
        
        max_val = max(1, 2**len(x_bits) - 1)
        x = (int(x_bits, 2) / max_val) * 2 - 1
        
        max_val = max(1, 2**len(y_bits) - 1)
        y = (int(y_bits, 2) / max_val) * 2 - 1
        
        return x, y
    
    def mcmc_step(self, current_x, current_y, step_size=0.1):
        """Perform one MCMC step with quantum or classical randomness"""
        # Generate random proposal
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
        mode_str = "Quantum" if self.quantum_mode else "Classical"
        print(f"Starting {mode_str} Monte Carlo Markov Chain π Estimation...")
        print(f"Using {self.n_qubits} bits, {self.n_iterations} iterations, {self.n_chains} chains")
        
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
        print("\n" + "="*60)
        print(f"{mode_str.upper()} MONTE CARLO MARKOV CHAIN RESULTS")
        print("="*60)
        print(f"Actual π:           {actual_pi:.8f}")
        print(f"Estimated π:        {final_pi_estimate:.8f}")
        print(f"Error:              {error:.8f}")
        print(f"Error percentage:   {error_percentage:.6f}%")
        print(f"Execution time:     {execution_time:.2f} seconds")
        print(f"Mode:              {mode_str} Random Number Generation")
        print(f"Per chain estimates: {[f'{est:.6f}' for est in pi_estimates_per_chain]}")
        print(f"Standard deviation: {np.std(pi_estimates_per_chain):.6f}")
        
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
            step = max(1, len(chain['x']) // 100)  # Show ~100 points per chain
            plt.plot(chain['x'][::step], chain['y'][::step], 'o-', 
                    color=color, alpha=0.7, markersize=2, linewidth=0.5,
                    label=f'Chain {i+1}')
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Markov Chain Paths')
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
        
        # Sample points for visualization if too many
        if len(all_x) > 5000:
            sample_indices = random.sample(range(len(all_x)), 5000)
            all_x = [all_x[i] for i in sample_indices]
            all_y = [all_y[i] for i in sample_indices]
            all_inside = [all_inside[i] for i in sample_indices]
        
        # Plot points inside and outside circle
        inside_x = [x for x, inside in zip(all_x, all_inside) if inside]
        inside_y = [y for y, inside in zip(all_y, all_inside) if inside]
        outside_x = [x for x, inside in zip(all_x, all_inside) if not inside]
        outside_y = [y for y, inside in zip(all_y, all_inside) if not inside]
        
        plt.scatter(inside_x, inside_y, c='red', s=1, alpha=0.6, label=f'Inside Circle ({len(inside_x)})')
        plt.scatter(outside_x, outside_y, c='blue', s=1, alpha=0.6, label=f'Outside Circle ({len(outside_x)})')
        plt.plot(circle_x, circle_y, 'k-', linewidth=2)
        
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('MCMC Sample Points Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 3. π Convergence
        plt.subplot(2, 3, 3)
        if self.pi_estimates:
            iterations = range(50, len(self.pi_estimates)*50 + 1, 50)
            plt.plot(iterations, self.pi_estimates, 'b-', linewidth=2, label='π Estimate')
            plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'True π = {np.pi:.6f}')
            plt.xlabel('Iterations')
            plt.ylabel('π Estimate')
            plt.title('π Estimation Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add final estimate annotation
            if len(self.pi_estimates) > 0:
                final_est = self.pi_estimates[-1]
                plt.annotate(f'Final: {final_est:.4f}', 
                           xy=(iterations[-1], final_est),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # 4. Chain Statistics
        plt.subplot(2, 3, 4)
        chain_pi_estimates = [chain['final_pi'] for chain in self.chain_histories]
        bars = plt.bar(range(1, self.n_chains + 1), chain_pi_estimates, 
                      color=colors, alpha=0.7)
        plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label=f'True π = {np.pi:.6f}')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, chain_pi_estimates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Chain Number')
        plt.ylabel('π Estimate')
        plt.title('π Estimates per Chain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Error Analysis
        plt.subplot(2, 3, 5)
        errors = [abs(est - np.pi) for est in chain_pi_estimates]
        error_bars = plt.bar(range(1, self.n_chains + 1), errors, color='orange', alpha=0.7)
        
        # Add error percentage labels
        for i, (bar, err) in enumerate(zip(error_bars, errors)):
            err_pct = (err / np.pi) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{err_pct:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Chain Number')
        plt.ylabel('Absolute Error')
        plt.title('Estimation Error per Chain')
        plt.grid(True, alpha=0.3)
        
        # 6. Method Information
        plt.subplot(2, 3, 6)
        mode_str = "Quantum" if self.quantum_mode else "Classical"
        
        info_text = f"""
{mode_str} Monte Carlo Markov Chain π Estimation

Method Details:
• Random Number Generation: {mode_str}
• Number of qubits/bits: {self.n_qubits}
• Iterations per chain: {self.n_iterations:,}
• Number of chains: {self.n_chains}
• Total samples: {self.n_iterations * self.n_chains:,}

Algorithm:
1. Generate random (x,y) coordinates
2. Use MCMC to explore the space [-1,1]²
3. Count points inside unit circle
4. Estimate π = 4 × (inside/total)

Statistics:
• Mean estimate: {np.mean(chain_pi_estimates):.6f}
• Standard deviation: {np.std(chain_pi_estimates):.6f}
• Best estimate: {max(chain_pi_estimates, key=lambda x: -abs(x-np.pi)):.6f}
• Worst estimate: {min(chain_pi_estimates, key=lambda x: -abs(x-np.pi)):.6f}
        """
        
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title(f'{mode_str} MCMC π Estimation Summary')
        
        plt.tight_layout()
        plt.show()

    def run_comparison(self):
        """Compare quantum vs classical if both are available"""
        if not QISKIT_AVAILABLE:
            print("Qiskit not available - cannot run quantum vs classical comparison")
            return
        
        print("Running Quantum vs Classical Comparison...")
        
        # Run quantum version
        self.quantum_mode = True
        q_pi, q_err, q_err_pct, q_time = self.run_estimation()
        q_chains = self.chain_histories.copy()
        
        # Reset for classical
        self.chain_histories = []
        self.pi_estimates = []
        
        # Run classical version
        self.quantum_mode = False
        c_pi, c_err, c_err_pct, c_time = self.run_estimation()
        c_chains = self.chain_histories.copy()
        
        # Comparison results
        print("\n" + "="*60)
        print("QUANTUM vs CLASSICAL COMPARISON")
        print("="*60)
        print(f"Quantum  π estimate: {q_pi:.8f} (error: {q_err_pct:.4f}%, time: {q_time:.2f}s)")
        print(f"Classical π estimate: {c_pi:.8f} (error: {c_err_pct:.4f}%, time: {c_time:.2f}s)")
        print(f"Difference: {abs(q_pi - c_pi):.8f}")
        print(f"Time ratio (Q/C): {q_time/c_time:.2f}")

# Run the estimation
if __name__ == "__main__":
    # Initialize estimator
    estimator = QuantumMCMC_PiEstimator(
        n_qubits=4,          # 4 qubits/bits for random number generation
        n_iterations=800,    # 800 iterations per chain
        n_chains=4           # 4 independent chains
    )
    
    # Run estimation
    pi_est, error, error_pct, exec_time = estimator.run_estimation()
    
    # Create visualizations
    estimator.visualize_results()
    
    # If Qiskit is available, you can also run a comparison
    # estimator.run_comparison()