import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit import QuantumCircuit, Aer, execute

class DashboardVisualizer:
    def __init__(self, chain_histories, pi_estimates, n_chains):
        self.chain_histories = chain_histories
        self.pi_estimates = pi_estimates
        self.n_chains = n_chains

    def display(self):
        sns.set(style="whitegrid")
        fig = plt.figure(figsize=(22, 14))

        # Colors
        colors = sns.color_palette("husl", self.n_chains)

        # 1. Markov Chain Paths
        ax1 = plt.subplot(2, 3, 1)
        theta = np.linspace(0, 2 * np.pi, 100)
        ax1.plot(np.cos(theta), np.sin(theta), 'k-', lw=2, label='Unit Circle')
        for i, (chain, color) in enumerate(zip(self.chain_histories, colors)):
            ax1.plot(chain['x'][::20], chain['y'][::20], 'o-', label=f'Chain {i+1}',
                     markersize=2, linewidth=0.5, alpha=0.7, color=color)
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Quantum Sampling Paths')
        ax1.legend(fontsize='small')
        ax1.set_aspect('equal')

        # 2. Sample Distribution
        ax2 = plt.subplot(2, 3, 2)
        all_x, all_y, all_inside = [], [], []
        for chain in self.chain_histories:
            all_x.extend(chain['x'][1:])
            all_y.extend(chain['y'][1:])
            all_inside.extend(chain['inside'])
        inside_x = [x for x, inside in zip(all_x, all_inside) if inside]
        inside_y = [y for y, inside in zip(all_y, all_inside) if inside]
        outside_x = [x for x, inside in zip(all_x, all_inside) if not inside]
        outside_y = [y for y, inside in zip(all_y, all_inside) if not inside]
        ax2.scatter(inside_x, inside_y, c='tomato', s=2, label='Inside Circle', alpha=0.6)
        ax2.scatter(outside_x, outside_y, c='royalblue', s=2, label='Outside Circle', alpha=0.6)
        ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=2)
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Quantum Sample Distribution')
        ax2.legend(fontsize='small')
        ax2.set_aspect('equal')

        # 3. π Convergence
        ax3 = plt.subplot(2, 3, 3)
        if self.pi_estimates:
            steps = list(range(50, len(self.pi_estimates) * 50 + 1, 50))
            ax3.plot(steps, self.pi_estimates, 'b-', lw=2, label='π Estimate')
            ax3.axhline(np.pi, color='red', linestyle='--', lw=2, label='True π')
            ax3.set_title('π Estimation Convergence')
            ax3.set_xlabel('Iterations')
            ax3.set_ylabel('π Estimate')
            ax3.legend()

        # 4. Estimates per Chain
        ax4 = plt.subplot(2, 3, 4)
        final_estimates = [c['final_pi'] for c in self.chain_histories]
        sns.barplot(x=list(range(1, self.n_chains + 1)), y=final_estimates,
                    palette=colors, ax=ax4)
        ax4.axhline(np.pi, color='red', linestyle='--', lw=2, label='True π')
        ax4.set_xlabel('Chain')
        ax4.set_ylabel('π Estimate')
        ax4.set_title('Final π Estimate (Quantum)')
        ax4.legend()

        # 5. Error per Chain
        ax5 = plt.subplot(2, 3, 5)
        errors = [abs(est - np.pi) for est in final_estimates]
        sns.barplot(x=list(range(1, self.n_chains + 1)), y=errors,
                    color='goldenrod', ax=ax5)
        ax5.set_xlabel('Chain')
        ax5.set_ylabel('Absolute Error')
        ax5.set_title('Estimation Error per Chain')

        # 6. Quantum Circuit Visualization
        ax6 = plt.subplot(2, 3, 6)
        qc = QuantumCircuit(4)
        qc.h(range(4))
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.measure_all()
        circuit_text = qc.draw(output='text')
        ax6.text(0.1, 0.95, "Quantum RNG Circuit:", fontsize=12, weight='bold')
        for i, line in enumerate(str(circuit_text).splitlines()):
            ax6.text(0, 0.9 - i * 0.06, line, family='monospace', fontsize=10)
        ax6.axis('off')
        ax6.set_title("Quantum Circuit Architecture")

        plt.tight_layout()
        plt.show()
