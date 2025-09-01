import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import colorsys

class SimulatedAnnealingPiEstimator:
    def __init__(self, num_agents=20, steps_per_agent=600, initial_temp=2.0, 
                 cooling_rate=0.995, min_temp=0.01):
        self.num_agents = num_agents
        self.steps_per_agent = steps_per_agent
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
        # Animation parameters
        self.trail_length = 60
        
        # Pi estimation tracking
        self.total_inside = 0
        self.total_points = 0
        self.pi_history = []
        self.temperature_history = []
        self.acceptance_history = []
        
        # Simulated annealing state
        self.current_temp = initial_temp
        self.step_count = 0
        
        # Multi-strategy agents
        self.agent_strategies = ['boundary_hunter', 'uncertainty_explorer', 'uniform_sampler', 'adaptive_local']
        
        self.initialize_agents()
    
    def initialize_agents(self):
        """Initialize agents with different strategies"""
        self.agents = []
        
        for i in range(self.num_agents):
            strategy = self.agent_strategies[i % len(self.agent_strategies)]
            
            # Strategic initial positioning
            if strategy == 'boundary_hunter':
                # Start near circle boundary
                angle = 2 * np.pi * i / self.num_agents
                x = 0.9 * np.cos(angle) + np.random.normal(0, 0.1)
                y = 0.9 * np.sin(angle) + np.random.normal(0, 0.1)
            elif strategy == 'uncertainty_explorer':
                # Start in corners (high-uncertainty regions)
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.8, 0.8)
            elif strategy == 'uniform_sampler':
                # Uniform distribution
                x = np.random.uniform(-0.9, 0.9)
                y = np.random.uniform(-0.9, 0.9)
            else:  # adaptive_local
                # Start in corners (high gradient regions)
                corner_signs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
                sign_x, sign_y = corner_signs[i % 4]
                x = sign_x * np.random.uniform(0.5, 0.9)
                y = sign_y * np.random.uniform(0.5, 0.9)
            
            # Clip to bounds
            x = np.clip(x, -0.98, 0.98)
            y = np.clip(y, -0.98, 0.98)
            
            agent = {
                'x': deque([x], maxlen=self.trail_length),
                'y': deque([y], maxlen=self.trail_length),
                'current_x': x,
                'current_y': y,
                'strategy': strategy,
                'energy': self.calculate_energy(x, y),
                'total_moves': 0,
                'accepted_moves': 0,
                'rejected_moves': 0,
                'best_energy': float('inf'),
                'local_samples': 0,
                'local_inside': 0,
                'efficiency': 0,
            }
            self.agents.append(agent)
    
    def is_inside_circle(self, x, y):
        """Check if point is inside unit circle"""
        return x**2 + y**2 <= 1.0
    
    def get_distance_to_boundary(self, x, y):
        """Get distance to circle boundary"""
        return abs(np.sqrt(x**2 + y**2) - 1.0)
    
    def calculate_energy(self, x, y):
        """Calculate energy (lower is better for boundary sampling)"""
        # Lower energy near the boundary (most informative for pi estimation)
        boundary_distance = self.get_distance_to_boundary(x, y)
        
        # Primary energy: distance to boundary (want to minimize)
        boundary_energy = boundary_distance
        
        # Secondary energy: slight bias toward less sampled regions
        # (This would require tracking sampling density, simplified here)
        
        return boundary_energy
    
    def get_current_temperature(self):
        """Get current temperature with proper cooling schedule"""
        return max(self.min_temp, self.initial_temp * (self.cooling_rate ** self.step_count))
    
    def acceptance_probability(self, old_energy, new_energy, temperature):
        """Calculate acceptance probability using Boltzmann distribution"""
        if new_energy < old_energy:
            # Always accept better solutions
            return 1.0
        else:
            # Accept worse solutions with probability based on temperature
            # CRITICAL: This creates the temperature-dependent acceptance behavior
            if temperature <= 0:
                return 0.0
            delta_energy = new_energy - old_energy
            return np.exp(-delta_energy / temperature)
    
    def simulated_annealing_move(self, agent, step):
        """Make simulated annealing move with proper temperature-dependent behavior"""
        current_temp = self.get_current_temperature()
        
        x_old = agent['current_x']
        y_old = agent['current_y']
        old_energy = agent['energy']
        
        # Step size based on temperature (high temp = large steps, low temp = small steps)
        temp_ratio = current_temp / self.initial_temp
        
        # HIGH TEMPERATURE: Random exploration with large steps
        # LOW TEMPERATURE: Focused exploitation with small steps
        if temp_ratio > 0.7:
            # High temperature: completely random exploration
            step_size = 0.3 + 0.4 * temp_ratio  # Large steps
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
        elif temp_ratio > 0.3:
            # Medium temperature: strategy-based with some randomness
            step_size = 0.1 + 0.2 * temp_ratio
            strategy = agent['strategy']
            
            if strategy == 'boundary_hunter':
                # Move toward boundary with noise
                current_dist = np.sqrt(x_old**2 + y_old**2)
                if current_dist > 0:
                    target_dist = 1.0
                    scale = target_dist / current_dist
                    target_x = x_old * scale
                    target_y = y_old * scale
                    dx = (target_x - x_old) * 0.3 + np.random.normal(0, step_size)
                    dy = (target_y - y_old) * 0.3 + np.random.normal(0, step_size)
                else:
                    dx = np.random.normal(0, step_size)
                    dy = np.random.normal(0, step_size)
            else:
                # Other strategies: mostly random at medium temp
                dx = np.random.normal(0, step_size)
                dy = np.random.normal(0, step_size)
        else:
            # Low temperature: precise local search
            step_size = 0.02 + 0.08 * temp_ratio  # Small steps
            
            # All strategies converge to boundary hunting at low temp
            boundary_dist = self.get_distance_to_boundary(x_old, y_old)
            current_dist = np.sqrt(x_old**2 + y_old**2)
            
            if current_dist > 0 and boundary_dist > 0.01:
                # Move toward boundary precisely
                direction_x = x_old / current_dist
                direction_y = y_old / current_dist
                
                if current_dist < 1.0:
                    # Inside: move outward toward boundary
                    dx = direction_x * step_size + np.random.normal(0, step_size * 0.2)
                    dy = direction_y * step_size + np.random.normal(0, step_size * 0.2)
                else:
                    # Outside: move inward toward boundary
                    dx = -direction_x * step_size + np.random.normal(0, step_size * 0.2)
                    dy = -direction_y * step_size + np.random.normal(0, step_size * 0.2)
            else:
                # Very close to boundary or at origin: tiny perturbations
                dx = np.random.normal(0, step_size * 0.5)
                dy = np.random.normal(0, step_size * 0.5)
        
        # Propose new position
        x_new = np.clip(x_old + dx, -0.98, 0.98)
        y_new = np.clip(y_old + dy, -0.98, 0.98)
        new_energy = self.calculate_energy(x_new, y_new)
        
        # Calculate acceptance probability (this is key for temperature behavior)
        accept_prob = self.acceptance_probability(old_energy, new_energy, current_temp)
        
        # Accept or reject based on probability
        agent['total_moves'] += 1
        
        if np.random.random() < accept_prob:
            # Accept the move
            agent['current_x'] = x_new
            agent['current_y'] = y_new
            agent['energy'] = new_energy
            agent['accepted_moves'] += 1
            
            if new_energy < agent['best_energy']:
                agent['best_energy'] = new_energy
        else:
            # Reject the move (stay at current position)
            agent['rejected_moves'] += 1
            # Note: we don't update position, so agent stays put
        
        # Store current position for animation (whether moved or not)
        agent['x'].append(agent['current_x'])
        agent['y'].append(agent['current_y'])
        
        # Update pi estimation tracking
        self.total_points += 1
        if self.is_inside_circle(agent['current_x'], agent['current_y']):
            self.total_inside += 1
            agent['local_inside'] += 1
        
        agent['local_samples'] += 1
        agent['efficiency'] = agent['local_inside'] / agent['local_samples'] if agent['local_samples'] > 0 else 0
        
        return agent['current_x'], agent['current_y'], accept_prob
    
    def get_current_pi_estimate(self):
        """Get current pi estimate"""
        if self.total_points == 0:
            return 0
        return 4 * self.total_inside / self.total_points
    
    def get_temperature_color(self, temperature_ratio):
        """Get color based on temperature: bright red (hot) to dark blue (cold)"""
        # temperature_ratio: 1.0 (hot) to 0.0 (cold)
        # Create smooth transition: Red -> Orange -> Yellow -> Green -> Blue
        
        if temperature_ratio > 0.8:
            # Very hot: bright red
            return (1.0, 0.0, 0.0)
        elif temperature_ratio > 0.6:
            # Hot: red to orange
            t = (temperature_ratio - 0.6) / 0.2
            return (1.0, 0.5 * (1.0 - t), 0.0)
        elif temperature_ratio > 0.4:
            # Medium-hot: orange to yellow
            t = (temperature_ratio - 0.4) / 0.2
            return (1.0, 0.5 + 0.5 * (1.0 - t), 0.0)
        elif temperature_ratio > 0.2:
            # Medium: yellow to green
            t = (temperature_ratio - 0.2) / 0.2
            return (1.0 * t, 1.0, 0.0)
        else:
            # Cold: green to blue
            t = temperature_ratio / 0.2
            return (0.0, 1.0 * t, 1.0)
    
    def get_accuracy_color(self, accuracy_percent):
        """Get smooth color transition from red to green based on accuracy"""
        # Normalize accuracy to 0-1 range (assume accuracy is between 0-100)
        # For pi estimation, we want smooth transition from ~90% to 100%
        
        # Define accuracy thresholds for smooth transition
        if accuracy_percent < 90:
            # Very low accuracy: pure red
            return (1.0, 0.0, 0.0)
        elif accuracy_percent >= 99.9:
            # Very high accuracy: pure green
            return (0.0, 1.0, 0.0)
        else:
            # Smooth transition from red to green
            # Map 90-99.9% accuracy to 0-1 range
            normalized = (accuracy_percent - 90) / 9.9
            normalized = np.clip(normalized, 0, 1)
            
            # Create smooth transition using HSV color space
            # Red (H=0) to Green (H=120) in HSV
            hue = normalized * 120 / 360  # Convert to 0-1 range for HSV
            saturation = 1.0
            value = 1.0
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            return (r, g, b)
    
    def create_animation(self):
        """Create enhanced animation with proper simulated annealing"""
        fig = plt.figure(figsize=(18, 12))
        
        # Main animation area
        ax_main = plt.subplot2grid((4, 5), (0, 0), colspan=3, rowspan=3)
        ax_main.set_xlim(-1.2, 1.2)
        ax_main.set_ylim(-1.2, 1.2)
        ax_main.set_aspect('equal')
        ax_main.set_facecolor('black')
        
        # Circle and square
        circle = plt.Circle((0, 0), 1.0, color='white', fill=False, linewidth=4, alpha=0.9)
        square = plt.Rectangle((-1, -1), 2, 2, color='white', fill=False, linewidth=2, alpha=0.7)
        ax_main.add_patch(circle)
        ax_main.add_patch(square)
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        
        # Dashboard panels
        ax_temperature = plt.subplot2grid((4, 5), (0, 3), colspan=2)
        ax_pi = plt.subplot2grid((4, 5), (1, 3), colspan=2)
        ax_acceptance = plt.subplot2grid((4, 5), (2, 3), colspan=2)
        ax_progress = plt.subplot2grid((4, 5), (3, 0), colspan=5)
        
        # Temperature display
        ax_temperature.set_xlim(0, 1)
        ax_temperature.set_ylim(0, 1)
        ax_temperature.set_title('Temperature', fontsize=12, fontweight='bold')
        temp_bar = ax_temperature.barh(0.5, 0, height=0.3, color='red')
        temp_text = ax_temperature.text(0.5, 0.5, '', ha='center', va='center', fontweight='bold')
        ax_temperature.set_xticks([])
        ax_temperature.set_yticks([])
        
        # Pi estimate with live updates
        ax_pi.set_xlim(0, 1)
        ax_pi.set_ylim(0, 1)
        ax_pi.set_title('π Estimate (Live)', fontsize=12, fontweight='bold')
        pi_text = ax_pi.text(0.5, 0.9, '', ha='center', va='center', fontsize=12, fontweight='bold')
        error_text = ax_pi.text(0.5, 0.7, '', ha='center', va='center', fontsize=10)
        accuracy_text = ax_pi.text(0.5, 0.5, '', ha='center', va='center', fontsize=10, fontweight='bold')
        samples_text = ax_pi.text(0.5, 0.3, '', ha='center', va='center', fontsize=9)
        convergence_text = ax_pi.text(0.5, 0.1, '', ha='center', va='center', fontsize=8)
        ax_pi.set_xticks([])
        ax_pi.set_yticks([])
        
        # Acceptance rates
        ax_acceptance.set_xlim(0, 1)
        ax_acceptance.set_ylim(0, 1)
        ax_acceptance.set_title('Acceptance Rate', fontsize=12, fontweight='bold')
        acceptance_text = ax_acceptance.text(0.5, 0.7, '', ha='center', va='center', fontsize=14, fontweight='bold')
        phase_text = ax_acceptance.text(0.5, 0.3, '', ha='center', va='center', fontsize=10)
        ax_acceptance.set_xticks([])
        ax_acceptance.set_yticks([])
        
        # Progress tracking
        ax_progress.set_title('Convergence Progress', fontsize=12, fontweight='bold')
        ax_progress.set_xlabel('Steps')
        ax_progress.set_ylabel('π Estimate')
        ax_progress.axhline(y=np.pi, color='gold', linestyle='--', linewidth=3, alpha=0.8)
        progress_line, = ax_progress.plot([], [], 'cyan', linewidth=2)
        ax_progress.set_facecolor('black')
        ax_progress.tick_params(colors='white')
        
        # Initialize agent plots - ALL AGENTS SAME COLOR (temperature-based)
        agent_trails = []
        current_points = []
        
        for i in range(self.num_agents):
            # ALL agents start with same initial color (hot red)
            initial_color = self.get_temperature_color(1.0)
            
            trail, = ax_main.plot([], [], '-', linewidth=2, alpha=0.7, color=initial_color)
            point, = ax_main.plot([], [], 'o', markersize=8, markeredgewidth=2, 
                                 markeredgecolor='white', color=initial_color)
            agent_trails.append(trail)
            current_points.append(point)
        
        # Reset simulation state
        self.total_inside = 0
        self.total_points = 0
        self.pi_history = []
        self.temperature_history = []
        self.acceptance_history = []
        self.step_count = 0
        self.current_temp = self.initial_temp
        self.initialize_agents()
        
        def animate(frame):
            if frame >= self.steps_per_agent:
                return
            
            self.step_count = frame
            current_temp = self.get_current_temperature()
            temp_ratio = current_temp / self.initial_temp
            
            # Update all agents
            total_acceptance = 0
            for i, agent in enumerate(self.agents):
                _, _, accept_prob = self.simulated_annealing_move(agent, frame)
                total_acceptance += accept_prob
            
            # Calculate actual acceptance rate from recent moves
            if frame > 0:
                recent_accepted = sum(a['accepted_moves'] for a in self.agents)
                recent_total = sum(a['total_moves'] for a in self.agents)
                actual_acceptance_rate = recent_accepted / recent_total if recent_total > 0 else 0
            else:
                actual_acceptance_rate = 1.0
            
            # Store history
            if frame % 3 == 0:
                current_pi = self.get_current_pi_estimate()
                self.pi_history.append(current_pi)
                self.temperature_history.append(current_temp)
                self.acceptance_history.append(actual_acceptance_rate)
            
            # ALL AGENTS USE SAME TEMPERATURE-BASED COLOR
            temp_color = self.get_temperature_color(temp_ratio)
            
            for i, agent in enumerate(self.agents):
                x_data = list(agent['x'])
                y_data = list(agent['y'])
                
                # ALL agents use same temperature-based color
                agent_trails[i].set_color(temp_color)
                agent_trails[i].set_data(x_data, y_data)
                
                # Update current point
                if len(x_data) > 0:
                    current_points[i].set_color(temp_color)
                    current_points[i].set_data([x_data[-1]], [y_data[-1]])
                    
                    # Marker size based on energy (lower energy = larger marker)
                    energy_ratio = max(0.1, 1.0 - agent['energy'])
                    marker_size = 6 + 6 * energy_ratio
                    current_points[i].set_markersize(marker_size)
            
            # Update dashboard with live data
            current_pi = self.get_current_pi_estimate()
            current_error = abs(current_pi - np.pi)
            
            # Calculate accuracy percentage (inverse of relative error)
            if current_pi != 0:
                relative_error = abs(current_pi - np.pi) / np.pi
                accuracy_percent = max(0, (1 - relative_error) * 100)
            else:
                accuracy_percent = 0
            
            # Temperature bar
            temp_bar[0].set_width(temp_ratio)
            temp_bar[0].set_color(temp_color)
            temp_text.set_text(f'{current_temp:.4f}')
            
            # Pi estimate with live updates
            pi_text.set_text(f'π = {current_pi:.6f}')
            error_text.set_text(f'Error: {current_error:.6f}')
            
            # Smooth color-coded accuracy using the new function
            accuracy_color = self.get_accuracy_color(accuracy_percent)
            accuracy_text.set_text(f'Accuracy: {accuracy_percent:.3f}%')
            accuracy_text.set_color(accuracy_color)
            
            samples_text.set_text(f'Samples: {self.total_points:,}')
            
            # Convergence trend
            if len(self.pi_history) > 10:
                recent_trend = np.mean(self.pi_history[-5:]) - np.mean(self.pi_history[-10:-5])
                if abs(recent_trend) < 0.001:
                    convergence_status = "CONVERGED"
                    convergence_color = 'green'
                elif recent_trend > 0:
                    convergence_status = "IMPROVING"
                    convergence_color = 'cyan'
                else:
                    convergence_status = "DECLINING"
                    convergence_color = 'orange'
                convergence_text.set_text(f'Trend: {convergence_status}')
                convergence_text.set_color(convergence_color)
            else:
                convergence_text.set_text('Trend: INITIALIZING')
                convergence_text.set_color('white')
            
            # Acceptance rates - show actual acceptance rate
            acceptance_text.set_text(f'{actual_acceptance_rate:.3f}')
            
            # Phase determination
            if temp_ratio > 0.6:
                phase = "EXPLORATION (High Temp)"
            elif temp_ratio > 0.2:
                phase = "TRANSITION (Med Temp)"
            else:
                phase = "EXPLOITATION (Low Temp)"
            
            phase_text.set_text(f'{phase}')
            
            # Progress plot
            if len(self.pi_history) > 1:
                steps = range(0, len(self.pi_history) * 3, 3)
                progress_line.set_data(steps, self.pi_history)
                ax_progress.relim()
                ax_progress.autoscale_view()
            
            # Title with temperature info
            progress_percent = (frame + 1) / self.steps_per_agent * 100
            title = f'Simulated Annealing π Estimation | Step {frame+1}/{self.steps_per_agent} | T={current_temp:.4f} | Accept={actual_acceptance_rate:.3f}'
            fig.suptitle(title, fontsize=16, fontweight='bold', color='white')
            
            return agent_trails + current_points + [progress_line]
        
        # Style the figure
        fig.patch.set_facecolor('black')
        for ax in [ax_temperature, ax_pi, ax_acceptance]:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
        
        # Set text colors for dashboard
        for text_obj in [pi_text, error_text, samples_text, convergence_text]:
            text_obj.set_color('white')
        
        for text_obj in [temp_text, acceptance_text, phase_text]:
            text_obj.set_color('white')
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=self.steps_per_agent, 
                           interval=50, blit=False, repeat=True)
        
        plt.tight_layout()
        return fig, anim

# Usage
if __name__ == "__main__":
    # Create simulated annealing estimator
    estimator = SimulatedAnnealingPiEstimator(
        num_agents=16,           # Multi-strategy agents
        steps_per_agent=1000,     # Steps per agent
        initial_temp=3.0,        # Higher initial temperature for more randomness
        cooling_rate=0.996,      # Slower cooling for gradual transition
        min_temp=0.01           # Minimum temperature
    )
    
    # Create and show animation
    fig, animation = estimator.create_animation()
    plt.show()
    
    # Print comprehensive statistics
    final_pi = estimator.get_current_pi_estimate()
    final_error = abs(final_pi - np.pi)
    final_accuracy = max(0, (1 - final_error / np.pi) * 100)
    
    print(f"\n=== SIMULATED ANNEALING RESULTS ===")
    print(f"π estimate: {final_pi:.10f}")
    print(f"True π: {np.pi:.10f}")
    print(f"Absolute error: {final_error:.10f}")
    print(f"Relative error: {final_error/np.pi * 100:.7f}%")
    print(f"Accuracy: {final_accuracy:.7f}%")
    print(f"Total points sampled: {estimator.total_points:,}")
    print(f"Points inside circle: {estimator.total_inside:,}")
    print(f"Final temperature: {estimator.get_current_temperature():.6f}")
    
    # Strategy analysis
    print(f"\n=== STRATEGY ANALYSIS ===")
    for strategy in estimator.agent_strategies:
        agents = [a for a in estimator.agents if a['strategy'] == strategy]
        if agents:
            avg_acceptance = np.mean([a['accepted_moves']/a['total_moves'] 
                                    if a['total_moves'] > 0 else 0 for a in agents])
            avg_energy = np.mean([a['energy'] for a in agents])
            print(f"{strategy}: Acceptance={avg_acceptance:.4f}, Final Energy={avg_energy:.4f}")