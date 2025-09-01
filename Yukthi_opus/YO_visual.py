import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import random
from collections import deque

class HybridMCMCSA:
    def __init__(self, n_chains=5, initial_temp=10.0, cooling_rate=0.995, reheat_prob=0.02):
        self.n_chains = n_chains
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reheat_prob = reheat_prob
        self.step = 0
        
        # Initialize chains with random positions and temperatures
        self.chains = []
        for i in range(n_chains):
            chain = {
                'position': np.array([random.uniform(-3, 3), random.uniform(-3, 3)]),
                'temperature': initial_temp * (0.5 + random.random()),
                'history': deque(maxlen=100),  # Keep more history for smoother trails
                'accepted_history': deque(maxlen=100),
                'energy': 0.0,
                'color': plt.cm.plasma(i / n_chains),
                'acceptance_count': 0,
                'total_proposals': 0,
                'acceptance_rate': 1.0,  # Start with high acceptance rate
                'recent_acceptances': deque(maxlen=50),  # Track recent acceptances for realistic rate
                'velocity': np.array([0.0, 0.0]),  # For momentum-based smoothing
                'energy_history': deque(maxlen=50)
            }
            self.chains.append(chain)
    
    def energy_2d(self, x, y):
        """Enhanced 2D landscape: multiple peaks and valleys with smoother transitions"""
        return (1.5 * np.sin(x) * np.cos(y) + 
                0.8 * np.sin(2*x + 1) * np.cos(2*y - 1) +
                0.5 * np.sin(3*x - 2) * np.cos(0.5*y + 1) +
                0.3 * np.exp(-((x-1)**2 + (y+1)**2)/2) +
                0.3 * np.exp(-((x+1.5)**2 + (y-1.5)**2)/3) +
                0.05 * (x**2 + y**2))
    
    def energy_3d(self, x, y, z):
        """Enhanced 3D landscape: volumetric energy function"""
        r = np.sqrt(x**2 + y**2 + z**2)
        return (np.sin(r) / (r + 0.1) + 
                0.4 * np.sin(2*x) * np.cos(y) * np.sin(z) +
                0.3 * np.exp(-((x-1)**2 + (y+1)**2 + z**2)/2) +
                0.05 * r**2)
    
    def gradient_2d(self, x, y, h=1e-4):
        """Numerical gradient for greedy step"""
        grad_x = (self.energy_2d(x + h, y) - self.energy_2d(x - h, y)) / (2 * h)
        grad_y = (self.energy_2d(x, y + h) - self.energy_2d(x, y - h)) / (2 * h)
        return np.array([grad_x, grad_y])
    
    def mcmc_proposal(self, chain, step_size=0.2):
        """MCMC random proposal step with momentum"""
        # Add momentum for smoother movement
        random_step = np.random.normal(0, step_size, 2)
        chain['velocity'] = 0.7 * chain['velocity'] + 0.3 * random_step
        proposal = chain['position'] + chain['velocity']
        return np.clip(proposal, -4, 4)
    
    def greedy_refinement(self, position, step_size=0.08):
        """Greedy local optimization step"""
        grad = self.gradient_2d(position[0], position[1])
        refined_pos = position - step_size * grad
        return np.clip(refined_pos, -4, 4)
    
    def sa_accept(self, current_energy, proposed_energy, temperature):
        """Simulated Annealing acceptance criterion"""
        if proposed_energy < current_energy:
            return True
        else:
            if temperature > 1e-10:  # Avoid division by zero
                prob = np.exp(-(proposed_energy - current_energy) / temperature)
                return random.random() < prob
            else:
                return False
    
    def update_temperature(self, chain):
        """Update temperature with cooling and occasional reheating"""
        chain['temperature'] *= self.cooling_rate
        
        # Occasional reheating with varying intensity
        if random.random() < self.reheat_prob:
            reheat_factor = 0.3 + 0.7 * random.random()
            chain['temperature'] = self.initial_temp * reheat_factor
    
    def step_chain(self, chain):
        """Perform one step of the hybrid algorithm"""
        # 1. MCMC Proposal
        proposal = self.mcmc_proposal(chain)
        
        # 2. Greedy Refinement
        refined_proposal = self.greedy_refinement(proposal)
        
        # 3. Energy evaluation
        current_energy = self.energy_2d(chain['position'][0], chain['position'][1])
        proposed_energy = self.energy_2d(refined_proposal[0], refined_proposal[1])
        
        # Store energy history
        chain['energy_history'].append(current_energy)
        
        # 4. SA Acceptance
        chain['total_proposals'] += 1
        accepted = self.sa_accept(current_energy, proposed_energy, chain['temperature'])
        
        if accepted:
            chain['position'] = refined_proposal
            chain['energy'] = proposed_energy
            chain['accepted_history'].append(refined_proposal.copy())
            chain['acceptance_count'] += 1
        
        # Track recent acceptances for realistic declining rate
        chain['recent_acceptances'].append(1 if accepted else 0)
        
        # Calculate acceptance rate based on recent history (shows realistic decline)
        if len(chain['recent_acceptances']) >= 10:
            chain['acceptance_rate'] = sum(chain['recent_acceptances']) / len(chain['recent_acceptances'])
        else:
            # For early steps, use cumulative rate
            chain['acceptance_rate'] = chain['acceptance_count'] / max(1, chain['total_proposals'])
        
        # Store in history for visualization
        chain['history'].append({
            'position': refined_proposal.copy(),
            'accepted': accepted,
            'step': self.step,
            'energy': proposed_energy
        })
        
        # 5. Temperature update
        self.update_temperature(chain)
        
        return accepted

class EnhancedVisualizer:
    def __init__(self, algorithm, mode='2d'):
        self.algo = algorithm
        self.mode = mode
        self.global_step_history = deque(maxlen=200)
        self.global_temp_history = deque(maxlen=200)
        self.global_acceptance_history = deque(maxlen=200)
        self.setup_plot()
        
    def setup_plot(self):
        if self.mode == '2d':
            # Create enhanced layout with subplots using constrained layout instead of tight_layout
            self.fig = plt.figure(figsize=(16, 12), constrained_layout=True)
            gs = GridSpec(3, 3, height_ratios=[2, 2, 0.8], width_ratios=[2, 2, 1], 
                         figure=self.fig)
            
            # Main landscape plot
            self.ax_main = self.fig.add_subplot(gs[:2, :2])
            
            # Temperature bars
            self.ax_temp = self.fig.add_subplot(gs[0, 2])
            
            # Acceptance rate bars  
            self.ax_accept = self.fig.add_subplot(gs[1, 2])
            
            # Statistics plots
            self.ax_stats = self.fig.add_subplot(gs[2, :])
            
            self.setup_2d_landscape()
            self.setup_statistics_plots()
        else:
            self.fig = plt.figure(figsize=(16, 10), constrained_layout=True)
            gs = GridSpec(2, 3, height_ratios=[3, 1], width_ratios=[2, 1, 1],
                         figure=self.fig)
            
            self.ax_main = self.fig.add_subplot(gs[0, 0], projection='3d')
            self.ax_temp = self.fig.add_subplot(gs[0, 1])
            self.ax_accept = self.fig.add_subplot(gs[0, 2])
            self.ax_stats = self.fig.add_subplot(gs[1, :])
            
            self.setup_3d_landscape()
            self.setup_statistics_plots()
    
    def setup_2d_landscape(self):
        """Setup enhanced 2D contour landscape"""
        x = np.linspace(-4, 4, 150)
        y = np.linspace(-4, 4, 150)
        X, Y = np.meshgrid(x, y)
        Z = self.algo.energy_2d(X, Y)
        
        # Create beautiful contour plot
        self.contour = self.ax_main.contour(X, Y, Z, levels=25, colors='white', 
                                           alpha=0.4, linewidths=0.5)
        self.contourf = self.ax_main.contourf(X, Y, Z, levels=30, cmap='terrain', alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(self.contourf, ax=self.ax_main, shrink=0.8)
        cbar.set_label('Energy', fontsize=12)
        
        self.ax_main.set_xlim(-4, 4)
        self.ax_main.set_ylim(-4, 4)
        self.ax_main.set_xlabel('X', fontsize=12)
        self.ax_main.set_ylabel('Y', fontsize=12)
        self.ax_main.set_title('Hybrid MCMC-Greedy-SA on Enhanced 2D Landscape', fontsize=14, pad=20)
        self.ax_main.grid(True, alpha=0.3)
        
        # Initialize enhanced chain plots
        self.chain_plots = []
        self.trail_plots = []
        self.rejected_plots = []
        self.energy_rings = []  # Energy level indicators
        
        for i, chain in enumerate(self.algo.chains):
            # Current position with glow effect
            plot, = self.ax_main.plot([], [], 'o', markersize=15, 
                                     color=chain['color'], label=f'Chain {i+1}',
                                     markeredgecolor='white', markeredgewidth=2)
            self.chain_plots.append(plot)
            
            # Accepted trail with gradient
            trail, = self.ax_main.plot([], [], '-', alpha=0.8, linewidth=3,
                                      color=chain['color'])
            self.trail_plots.append(trail)
            
            # Rejected moves
            rejected, = self.ax_main.plot([], [], 'x', markersize=6, alpha=0.2,
                                         color=chain['color'], markeredgewidth=2)
            self.rejected_plots.append(rejected)
            
            # Energy level ring
            energy_ring = plt.Circle((0, 0), 0.1, fill=False, 
                                   color=chain['color'], alpha=0.3, linewidth=2)
            self.ax_main.add_patch(energy_ring)
            self.energy_rings.append(energy_ring)
    
    def setup_3d_landscape(self):
        """Setup enhanced 3D scatter landscape"""
        # Create a denser sample of the 3D energy landscape
        n_points = 2000
        x_sample = np.random.uniform(-3, 3, n_points)
        y_sample = np.random.uniform(-3, 3, n_points)
        z_sample = np.random.uniform(-3, 3, n_points)
        energies = self.algo.energy_3d(x_sample, y_sample, z_sample)
        
        # Plot landscape points colored by energy with better alpha
        self.landscape_scatter = self.ax_main.scatter(x_sample, y_sample, z_sample, 
                                                     c=energies, alpha=0.15, s=20, 
                                                     cmap='terrain')
        
        self.ax_main.set_xlim(-4, 4)
        self.ax_main.set_ylim(-4, 4)
        self.ax_main.set_zlim(-4, 4)
        self.ax_main.set_xlabel('X', fontsize=12)
        self.ax_main.set_ylabel('Y', fontsize=12)
        self.ax_main.set_zlabel('Z', fontsize=12)
        self.ax_main.set_title('Hybrid MCMC-Greedy-SA on Enhanced 3D Landscape', fontsize=14)
        
        # Initialize enhanced chain plots for 3D
        self.chain_plots = []
        self.trail_plots = []
        self.current_scatters = [None] * len(self.algo.chains)
        
        for i, chain in enumerate(self.algo.chains):
            # Add z-coordinate to chains for 3D
            if len(chain['position']) == 2:
                chain['position'] = np.append(chain['position'], random.uniform(-2, 2))
                chain['velocity'] = np.array([0.0, 0.0, 0.0])
            
            # Trail
            trail, = self.ax_main.plot([], [], [], '-', alpha=0.8, linewidth=3,
                                      color=chain['color'])
            self.trail_plots.append(trail)
    
    def setup_statistics_plots(self):
        """Setup temperature and acceptance rate visualization"""
        # Temperature bars
        self.ax_temp.set_title('Chain Temperatures', fontsize=12)
        self.ax_temp.set_ylim(0, self.algo.initial_temp * 1.1)
        self.ax_temp.set_xlim(-0.5, len(self.algo.chains) - 0.5)
        self.ax_temp.set_xlabel('Chain ID')
        self.ax_temp.set_ylabel('Temperature')
        self.ax_temp.grid(True, alpha=0.3)
        
        self.temp_bars = []
        for i in range(len(self.algo.chains)):
            bar = self.ax_temp.bar(i, 0, color=self.algo.chains[i]['color'], alpha=0.7)[0]
            self.temp_bars.append(bar)
        
        # Acceptance rate bars
        self.ax_accept.set_title('Acceptance Rates', fontsize=12)
        self.ax_accept.set_ylim(0, 1.0)
        self.ax_accept.set_xlim(-0.5, len(self.algo.chains) - 0.5)
        self.ax_accept.set_xlabel('Chain ID')
        self.ax_accept.set_ylabel('Rate')
        self.ax_accept.grid(True, alpha=0.3)
        
        self.accept_bars = []
        for i in range(len(self.algo.chains)):
            bar = self.ax_accept.bar(i, 0, color=self.algo.chains[i]['color'], alpha=0.7)[0]
            self.accept_bars.append(bar)
        
        # Statistics time series
        self.ax_stats.set_title('Live Statistics', fontsize=12)
        self.ax_stats.set_xlabel('Step')
        self.ax_stats.set_ylabel('Value')
        self.ax_stats.grid(True, alpha=0.3)
        
        self.temp_line, = self.ax_stats.plot([], [], 'r-', label='Avg Temperature', linewidth=2)
        self.accept_line, = self.ax_stats.plot([], [], 'g-', label='Avg Acceptance', linewidth=2)
        self.ax_stats.legend()
        self.ax_stats.set_ylim(0, 1)
    
    def update_temperature_color(self, chain_idx):
        """Update chain color based on temperature with smooth transitions"""
        chain = self.algo.chains[chain_idx]
        # Normalize temperature for color mapping
        temp_norm = np.clip(chain['temperature'] / self.algo.initial_temp, 0, 1)
        # Create smooth color transition from blue (cool) to red (hot)
        color = plt.cm.plasma(temp_norm)
        return color
    
    def animate_2d(self, frame):
        """Enhanced 2D animation function"""
        # Perform multiple steps per frame for smoother animation
        for _ in range(2):
            for chain in self.algo.chains:
                self.algo.step_chain(chain)
            self.algo.step += 1
        
        # Update main visualization
        for i, (chain, chain_plot, trail_plot, rejected_plot, energy_ring) in enumerate(
            zip(self.algo.chains, self.chain_plots, self.trail_plots, 
                self.rejected_plots, self.energy_rings)):
            
            # Update current position with temperature-based effects
            temp_color = self.update_temperature_color(i)
            temp_size = 12 + 8 * (chain['temperature'] / self.algo.initial_temp)
            
            chain_plot.set_data([chain['position'][0]], [chain['position'][1]])
            chain_plot.set_color(temp_color)
            chain_plot.set_markersize(temp_size)
            
            # Update accepted trail with fading effect
            if len(chain['accepted_history']) > 1:
                positions = np.array(list(chain['accepted_history']))
                trail_plot.set_data(positions[:, 0], positions[:, 1])
                trail_plot.set_color(temp_color)
                # Vary trail width based on recent acceptance
                trail_width = 2 + 3 * chain['acceptance_rate']
                trail_plot.set_linewidth(trail_width)
            
            # Update rejected moves with better fading
            if chain['history']:
                rejected_pos = []
                current_step = self.algo.step
                
                for move in list(chain['history'])[-30:]:
                    if not move['accepted']:
                        rejected_pos.append(move['position'])
                
                if rejected_pos:
                    rejected_pos = np.array(rejected_pos)
                    rejected_plot.set_data(rejected_pos[:, 0], rejected_pos[:, 1])
                    # Fade rejected points based on temperature
                    alpha = 0.1 + 0.3 * (chain['temperature'] / self.algo.initial_temp)
                    rejected_plot.set_alpha(alpha)
            
            # Update energy ring
            energy_ring.center = (chain['position'][0], chain['position'][1])
            ring_radius = 0.1 + 0.3 * (chain['temperature'] / self.algo.initial_temp)
            energy_ring.set_radius(ring_radius)
            energy_ring.set_color(temp_color)
        
        self.update_statistics()
        
        return (self.chain_plots + self.trail_plots + self.rejected_plots + 
                self.temp_bars + self.accept_bars + [self.temp_line, self.accept_line])
    
    def animate_3d(self, frame):
        """Enhanced 3D animation function"""
        # Perform algorithm steps for 3D
        for _ in range(2):
            for chain in self.algo.chains:
                # 3D version of the algorithm with momentum
                if len(chain['velocity']) != 3:
                    chain['velocity'] = np.array([0.0, 0.0, 0.0])
                
                random_step = np.random.normal(0, 0.2, 3)
                chain['velocity'] = 0.7 * chain['velocity'] + 0.3 * random_step
                proposal = chain['position'] + chain['velocity']
                proposal = np.clip(proposal, -4, 4)
                
                current_energy = self.algo.energy_3d(chain['position'][0], 
                                                   chain['position'][1], 
                                                   chain['position'][2])
                proposed_energy = self.algo.energy_3d(proposal[0], proposal[1], proposal[2])
                
                chain['total_proposals'] += 1
                if self.algo.sa_accept(current_energy, proposed_energy, chain['temperature']):
                    chain['position'] = proposal
                    chain['accepted_history'].append(proposal.copy())
                    chain['acceptance_count'] += 1
                    accepted = True
                else:
                    accepted = False
                
                # Track recent acceptances for realistic declining rate
                if 'recent_acceptances' not in chain:
                    chain['recent_acceptances'] = deque(maxlen=50)
                chain['recent_acceptances'].append(1 if accepted else 0)
                
                # Calculate acceptance rate based on recent history
                if len(chain['recent_acceptances']) >= 10:
                    chain['acceptance_rate'] = sum(chain['recent_acceptances']) / len(chain['recent_acceptances'])
                else:
                    chain['acceptance_rate'] = chain['acceptance_count'] / max(1, chain['total_proposals'])
                self.algo.update_temperature(chain)
            
            self.algo.step += 1
        
        # Update 3D visualizations
        for i, (chain, trail_plot) in enumerate(zip(self.algo.chains, self.trail_plots)):
            temp_color = self.update_temperature_color(i)
            
            # Clear previous scatter point
            if self.current_scatters[i] is not None:
                self.current_scatters[i].remove()
            
            size = 60 + 80 * (chain['temperature'] / self.algo.initial_temp)
            self.current_scatters[i] = self.ax_main.scatter(
                [chain['position'][0]], [chain['position'][1]], [chain['position'][2]], 
                s=size, color=temp_color, alpha=0.9, edgecolors='white', linewidth=2)
            
            # Update trail with better visualization
            if len(chain['accepted_history']) > 1:
                positions = np.array(list(chain['accepted_history']))
                trail_plot.set_data(positions[:, 0], positions[:, 1])
                trail_plot.set_3d_properties(positions[:, 2])
                trail_plot.set_color(temp_color)
                trail_plot.set_linewidth(2 + 3 * chain['acceptance_rate'])
        
        self.update_statistics()
        
        return self.trail_plots + [self.temp_line, self.accept_line]
    
    def update_statistics(self):
        """Update all statistics displays"""
        # Update temperature bars
        for i, (chain, bar) in enumerate(zip(self.algo.chains, self.temp_bars)):
            bar.set_height(chain['temperature'])
            temp_color = self.update_temperature_color(i)
            bar.set_color(temp_color)
            bar.set_alpha(0.7)
        
        # Update acceptance rate bars
        for i, (chain, bar) in enumerate(zip(self.algo.chains, self.accept_bars)):
            bar.set_height(chain['acceptance_rate'])
            bar.set_color(self.algo.chains[i]['color'])
            bar.set_alpha(0.7)
        
        # Update time series
        avg_temp = np.mean([chain['temperature'] for chain in self.algo.chains]) / self.algo.initial_temp
        avg_accept = np.mean([chain['acceptance_rate'] for chain in self.algo.chains])
        
        self.global_step_history.append(self.algo.step)
        self.global_temp_history.append(avg_temp)
        self.global_acceptance_history.append(avg_accept)
        
        if len(self.global_step_history) > 1:
            self.temp_line.set_data(list(self.global_step_history), list(self.global_temp_history))
            self.accept_line.set_data(list(self.global_step_history), list(self.global_acceptance_history))
            
            # Auto-scale x-axis
            self.ax_stats.set_xlim(max(0, self.algo.step - 200), self.algo.step + 10)
    
    def start_animation(self, interval=50):
        """Start the enhanced animation"""
        if self.mode == '2d':
            anim_func = self.animate_2d
        else:
            anim_func = self.animate_3d
        
        self.animation = FuncAnimation(self.fig, anim_func, interval=interval, 
                                     blit=False, cache_frame_data=False)
        
        if self.mode == '2d':
            self.ax_main.legend(loc='upper right', framealpha=0.9)
        
        # Use constrained layout instead of tight_layout to avoid warnings
        plt.show()

def main():
    """Enhanced main function"""
    print("🔥 Enhanced Hybrid MCMC-Greedy-SA Visualization")
    print("=" * 50)
    print("Choose visualization mode:")
    print("1. 2D Enhanced Landscape (recommended)")
    print("2. 3D Enhanced Landscape")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        mode = '3d' if choice == '2' else '2d'
    except:
        mode = '2d'
    
    print(f"\n🚀 Starting {mode.upper()} enhanced visualization...")
    print("Features:")
    print("- Live temperature and acceptance rate bars")
    print("- Smooth momentum-based movement")
    print("- Enhanced visual effects and colors")
    print("- Real-time statistics")
    print("- Multiple steps per frame for smoother animation")
    print("\nClose the plot window to exit.")
    
    # Create enhanced algorithm instance
    algorithm = HybridMCMCSA(n_chains=5, initial_temp=8.0, 
                            cooling_rate=0.996, reheat_prob=0.025)
    
    # Create and start enhanced visualization
    visualizer = EnhancedVisualizer(algorithm, mode=mode)
    visualizer.start_animation(interval=30)  # Faster updates for smoother animation

if __name__ == "__main__":
    main()