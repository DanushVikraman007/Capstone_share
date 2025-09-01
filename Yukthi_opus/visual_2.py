import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import colorsys

class HybridMCMCGreedySAOptimizer:
    def __init__(self, bounds=(-4, 4), initial_temp=3.0, cooling_rate=0.997, 
                 min_temp=0.005, max_iterations=800, tabu_size=15, 
                 reheating_threshold=40, mcmc_step_size=0.4):
        
        # Core parameters
        self.bounds = bounds
        self.dimension = 2
        self.max_iterations = max_iterations
        
        # SA parameters
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.current_temp = initial_temp
        
        # Algorithm parameters
        self.mcmc_step_size = mcmc_step_size
        self.greedy_samples = 12
        self.greedy_radius = 0.25
        
        # Memory and adaptivity
        self.tabu_size = tabu_size
        self.tabu_list = deque(maxlen=tabu_size)
        self.reheating_threshold = reheating_threshold
        self.stagnation_count = 0
        
        # Animation parameters
        self.trail_length = 80
        
        # State tracking
        self.iteration = 0
        self.acceptance_window = deque(maxlen=50)
        self.best_history = []
        self.temp_history = []
        self.acceptance_history = []
        self.reheating_events = []
        
        # Solutions
        self.current_solution = self.random_solution()
        self.current_value = self.rastrigin_function(self.current_solution)
        self.best_solution = self.current_solution.copy()
        self.best_value = self.current_value
        
        # Animation state
        self.proposed_solution = self.current_solution.copy()
        self.greedy_solution = self.current_solution.copy()
        self.last_accepted = False
        self.last_reheating = False
        self.reheating_intensity = 0.0
        
        # Trajectory for smooth trails
        self.solution_trail = deque(maxlen=self.trail_length)
        self.best_trail = deque(maxlen=self.trail_length)
        self.proposed_trail = deque(maxlen=20)
        self.greedy_trail = deque(maxlen=20)
        
        # Solution cache
        self.solution_cache = {}
        
        # Initialize trails
        for _ in range(5):
            self.solution_trail.append(self.current_solution.copy())
            self.best_trail.append(self.best_solution.copy())
    
    def rastrigin_function(self, x):
        """Smooth Rastrigin function - multimodal optimization landscape"""
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x)
    
    def random_solution(self):
        """Generate random solution within bounds"""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dimension)
    
    def evaluate_cached(self, solution):
        """Cached function evaluation"""
        key = tuple(np.round(solution, 5))
        if key not in self.solution_cache:
            self.solution_cache[key] = self.rastrigin_function(solution)
        return self.solution_cache[key]
    
    def is_tabu(self, solution, tolerance=0.15):
        """Check if solution is in tabu memory"""
        for tabu_sol in self.tabu_list:
            if np.linalg.norm(solution - tabu_sol) < tolerance:
                return True
        return False
    
    def mcmc_proposal(self, current):
        """MCMC Gaussian proposal with adaptive step size"""
        # Adaptive step size based on temperature
        temp_ratio = self.current_temp / self.initial_temp
        adaptive_step = self.mcmc_step_size * (0.5 + 1.5 * temp_ratio)
        
        proposal = current + np.random.normal(0, adaptive_step, self.dimension)
        # Soft boundary handling with reflection
        for i in range(len(proposal)):
            if proposal[i] < self.bounds[0]:
                proposal[i] = self.bounds[0] + (self.bounds[0] - proposal[i])
            elif proposal[i] > self.bounds[1]:
                proposal[i] = self.bounds[1] - (proposal[i] - self.bounds[1])
        
        proposal = np.clip(proposal, self.bounds[0], self.bounds[1])
        return proposal
    
    def greedy_refinement(self, solution):
        """Intelligent local greedy search"""
        best_local = solution.copy()
        best_local_value = self.evaluate_cached(solution)
        
        # Temperature-dependent greedy intensity
        temp_ratio = self.current_temp / self.initial_temp
        search_radius = self.greedy_radius * (0.3 + 0.7 * (1 - temp_ratio))
        samples = max(4, int(self.greedy_samples * (1 - temp_ratio * 0.5)))
        
        # Systematic + random search
        angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
        
        for angle in angles:
            # Radial search at different distances
            for radius_scale in [0.3, 0.7, 1.0]:
                radius = search_radius * radius_scale
                
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                
                candidate = solution + np.array([dx, dy])
                candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
                
                if not self.is_tabu(candidate):
                    candidate_value = self.evaluate_cached(candidate)
                    
                    if candidate_value < best_local_value:
                        best_local = candidate.copy()
                        best_local_value = candidate_value
        
        return best_local, best_local_value
    
    def acceptance_probability(self, current_value, new_value):
        """Metropolis acceptance with numerical stability"""
        if new_value <= current_value:
            return 1.0
        elif self.current_temp <= 0:
            return 0.0
        else:
            delta = (new_value - current_value) / self.current_temp
            return np.exp(-min(delta, 50))  # Prevent overflow
    
    def update_temperature(self):
        """Advanced temperature schedule with reheating"""
        # Standard exponential cooling
        self.current_temp = max(self.min_temp, 
                               self.current_temp * self.cooling_rate)
        
        # Stagnation detection and reheating
        reheating_occurred = False
        if (self.iteration > self.reheating_threshold and 
            len(self.best_history) >= self.reheating_threshold):
            
            recent_window = self.best_history[-self.reheating_threshold:]
            improvement = recent_window[0] - recent_window[-1]
            
            # Trigger reheating if insufficient improvement
            if improvement < 1e-5:
                # Adaptive reheating intensity
                reheat_factor = min(8.0, 2.0 + self.stagnation_count * 0.5)
                self.current_temp = min(self.initial_temp * 0.6, 
                                      self.current_temp * reheat_factor)
                self.reheating_events.append(self.iteration)
                self.stagnation_count += 1
                reheating_occurred = True
        
        return reheating_occurred
    
    def step(self):
        """Execute one iteration of the hybrid algorithm"""
        # Store previous state
        prev_solution = self.current_solution.copy()
        
        # 1. MCMC Proposal
        self.proposed_solution = self.mcmc_proposal(self.current_solution)
        proposed_value = self.evaluate_cached(self.proposed_solution)
        
        # 2. Greedy Refinement
        self.greedy_solution, greedy_value = self.greedy_refinement(self.proposed_solution)
        
        # 3. SA Acceptance Decision
        accept_prob = self.acceptance_probability(self.current_value, greedy_value)
        accepted = (np.random.random() < accept_prob and 
                   not self.is_tabu(self.greedy_solution))
        
        # 4. Update state
        if accepted:
            self.current_solution = self.greedy_solution.copy()
            self.current_value = greedy_value
            self.tabu_list.append(self.current_solution.copy())
            self.last_accepted = True
            
            # Track global best
            if greedy_value < self.best_value:
                self.best_solution = self.current_solution.copy()
                self.best_value = greedy_value
        else:
            self.last_accepted = False
        
        # 5. Update trails for smooth animation
        self.solution_trail.append(self.current_solution.copy())
        self.best_trail.append(self.best_solution.copy())
        self.proposed_trail.append(self.proposed_solution.copy())
        self.greedy_trail.append(self.greedy_solution.copy())
        
        # 6. Update tracking
        self.acceptance_window.append(1 if accepted else 0)
        self.best_history.append(self.best_value)
        
        # 7. Temperature update and reheating
        self.last_reheating = self.update_temperature()
        if self.last_reheating:
            self.reheating_intensity = 1.0
        else:
            self.reheating_intensity = max(0, self.reheating_intensity - 0.05)
        
        self.temp_history.append(self.current_temp)
        
        # Calculate sliding window acceptance rate
        if len(self.acceptance_window) > 0:
            acceptance_rate = np.mean(self.acceptance_window)
        else:
            acceptance_rate = 0
        self.acceptance_history.append(acceptance_rate)
        
        self.iteration += 1
        
        return accepted, accept_prob
    
    def is_converged(self, tolerance=1e-6, window=80):
        """Check for convergence"""
        if len(self.best_history) < window:
            return False
        recent_best = self.best_history[-window:]
        return max(recent_best) - min(recent_best) < tolerance

class SmoothHybridVisualizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
        # Create elegant figure layout
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Main landscape (larger, more prominent)
        self.ax_main = plt.subplot2grid((5, 6), (0, 0), colspan=4, rowspan=4)
        
        # Compact dashboard
        self.ax_stats = plt.subplot2grid((5, 6), (0, 4), colspan=2, rowspan=2)
        self.ax_temp = plt.subplot2grid((5, 6), (2, 4), colspan=1)
        self.ax_accept = plt.subplot2grid((5, 6), (2, 5), colspan=1)
        self.ax_phase = plt.subplot2grid((5, 6), (3, 4), colspan=2)
        self.ax_progress = plt.subplot2grid((5, 6), (4, 0), colspan=6)
        
        self.setup_landscape()
        self.setup_dashboard()
        
    def setup_landscape(self):
        """Create beautiful objective function landscape"""
        # High-resolution mesh for smooth visualization
        resolution = 120
        x_range = np.linspace(self.optimizer.bounds[0], self.optimizer.bounds[1], resolution)
        y_range = np.linspace(self.optimizer.bounds[0], self.optimizer.bounds[1], resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate landscape
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.optimizer.rastrigin_function([X[i, j], Y[i, j]])
        
        # Beautiful contour visualization
        levels = np.logspace(0, 2, 25)  # Logarithmic levels for better detail
        self.ax_main.contourf(X, Y, Z, levels=levels, cmap='plasma', alpha=0.9)
        contours = self.ax_main.contour(X, Y, Z, levels=levels[::3], 
                                       colors='white', alpha=0.4, linewidths=0.8)
        
        # Mark global minimum
        self.ax_main.plot(0, 0, 'w*', markersize=15, markeredgecolor='gold', 
                         markeredgewidth=2, label='Global Minimum', zorder=10)
        
        # Setup axes
        self.ax_main.set_xlim(self.optimizer.bounds[0], self.optimizer.bounds[1])
        self.ax_main.set_ylim(self.optimizer.bounds[0], self.optimizer.bounds[1])
        self.ax_main.set_aspect('equal')
        self.ax_main.set_facecolor('#0a0a0a')
        self.ax_main.tick_params(colors='white', labelsize=10)
        self.ax_main.set_xlabel('X', color='white', fontsize=12)
        self.ax_main.set_ylabel('Y', color='white', fontsize=12)
        
        # Initialize smooth trajectory lines
        self.main_trail, = self.ax_main.plot([], [], '-', linewidth=3, alpha=0.8, 
                                           color='cyan', label='Search Trail')
        self.best_trail, = self.ax_main.plot([], [], '-', linewidth=4, alpha=0.9, 
                                           color='gold', label='Best Trail')
        
        # Initialize markers with glow effects
        self.current_point, = self.ax_main.plot([], [], 'o', markersize=14, 
                                              markeredgewidth=3, zorder=8)
        self.proposed_point, = self.ax_main.plot([], [], 's', markersize=12, 
                                                color='orange', markeredgecolor='white', 
                                                markeredgewidth=2, alpha=0.9, zorder=7)
        self.greedy_point, = self.ax_main.plot([], [], '^', markersize=12, 
                                             color='lime', markeredgecolor='white', 
                                             markeredgewidth=2, alpha=0.9, zorder=7)
        self.best_point, = self.ax_main.plot([], [], '*', markersize=18, 
                                           color='gold', markeredgecolor='red', 
                                           markeredgewidth=3, zorder=9)
        
        # Reheating effect circle
        self.reheating_circle = plt.Circle((0, 0), 0.3, color='red', fill=False, 
                                         linewidth=6, alpha=0.0)
        self.ax_main.add_patch(self.reheating_circle)
        
        # Legend
        self.ax_main.legend(loc='upper right', facecolor='white', 
                          edgecolor='white', fontsize=10)
    
    def setup_dashboard(self):
        """Setup sleek dashboard panels"""
        dashboard_axes = [self.ax_stats, self.ax_temp, self.ax_accept, 
                         self.ax_phase, self.ax_progress]
        
        for ax in dashboard_axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
        
        # Stats panel
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.set_title('Live Statistics', fontsize=14, fontweight='bold', color='white')
        self.ax_stats.set_xticks([])
        self.ax_stats.set_yticks([])
        
        # Temperature panel
        self.ax_temp.set_xlim(0, 1)
        self.ax_temp.set_ylim(0, 1)
        self.ax_temp.set_title('Temperature', fontsize=12, fontweight='bold', color='white')
        self.temp_bar = self.ax_temp.barh(0.3, 0, height=0.4, alpha=0.9)
        self.ax_temp.set_xticks([])
        self.ax_temp.set_yticks([])
        
        # Acceptance panel
        self.ax_accept.set_xlim(0, 1)
        self.ax_accept.set_ylim(0, 1)
        self.ax_accept.set_title('Acceptance', fontsize=12, fontweight='bold', color='white')
        self.accept_bar = self.ax_accept.barh(0.3, 0, height=0.4, color='cyan', alpha=0.9)
        self.ax_accept.set_xticks([])
        self.ax_accept.set_yticks([])
        
        # Phase indicator
        self.ax_phase.set_xlim(0, 1)
        self.ax_phase.set_ylim(0, 1)
        self.ax_phase.set_title('Algorithm Phase', fontsize=12, fontweight='bold', color='white')
        self.ax_phase.set_xticks([])
        self.ax_phase.set_yticks([])
        
        # Progress tracking
        self.ax_progress.set_title('Convergence Progress', fontsize=14, fontweight='bold', color='white')
        self.ax_progress.set_xlabel('Iteration', color='white', fontsize=12)
        self.ax_progress.set_ylabel('Best Value (log scale)', color='white', fontsize=12)
        self.ax_progress.set_yscale('log')
        self.progress_line, = self.ax_progress.plot([], [], 'gold', linewidth=3, alpha=0.9)
        self.ax_progress.grid(True, alpha=0.3, color='white')
        
        # Text elements with better positioning
        self.iter_text = self.ax_stats.text(0.05, 0.85, '', fontsize=13, color='white', fontweight='bold')
        self.best_text = self.ax_stats.text(0.05, 0.70, '', fontsize=12, color='lime', fontweight='bold')
        self.current_text = self.ax_stats.text(0.05, 0.55, '', fontsize=11, color='cyan')
        self.improvement_text = self.ax_stats.text(0.05, 0.40, '', fontsize=11, color='orange')
        self.cache_text = self.ax_stats.text(0.05, 0.25, '', fontsize=10, color='lightgray')
        self.tabu_text = self.ax_stats.text(0.05, 0.10, '', fontsize=10, color='lightgray')
        
        self.temp_value_text = self.ax_temp.text(0.5, 0.1, '', ha='center', va='center', 
                                               fontsize=11, color='white', fontweight='bold')
        self.accept_value_text = self.ax_accept.text(0.5, 0.1, '', ha='center', va='center', 
                                                   fontsize=11, color='white', fontweight='bold')
        
        self.phase_main_text = self.ax_phase.text(0.5, 0.7, '', ha='center', va='center', 
                                                fontsize=13, color='white', fontweight='bold')
        self.phase_detail_text = self.ax_phase.text(0.5, 0.4, '', ha='center', va='center', 
                                                  fontsize=10, color='lightgray')
        self.reheating_text = self.ax_phase.text(0.5, 0.1, '', ha='center', va='center', 
                                               fontsize=11, color='red', fontweight='bold')
    
    def get_smooth_colors(self, temp_ratio):
        """Smooth color transitions based on temperature"""
        # Use HSV for smoother color transitions
        if temp_ratio > 0.8:
            # Very hot: bright red-orange
            hue = 0.05  # Orange-red
            sat = 1.0
            val = 1.0
        elif temp_ratio > 0.6:
            # Hot: orange to yellow
            hue = 0.1 + 0.05 * (temp_ratio - 0.6) / 0.2
            sat = 1.0
            val = 1.0
        elif temp_ratio > 0.4:
            # Medium: yellow to green
            hue = 0.15 + 0.15 * (temp_ratio - 0.4) / 0.2
            sat = 0.9
            val = 1.0
        elif temp_ratio > 0.2:
            # Cool: green to cyan
            hue = 0.3 + 0.2 * (temp_ratio - 0.2) / 0.2
            sat = 0.8
            val = 0.9
        else:
            # Cold: cyan to blue
            hue = 0.5 + 0.15 * temp_ratio / 0.2
            sat = 0.9
            val = 0.8
        
        return colorsys.hsv_to_rgb(hue, sat, val)
    
    def create_animation(self):
        """Create the smooth animation"""
        def animate(frame):
            if frame >= self.optimizer.max_iterations:
                return
            
            # Perform optimization step
            accepted, accept_prob = self.optimizer.step()
            
            # Calculate smooth metrics
            temp_ratio = self.optimizer.current_temp / self.optimizer.initial_temp
            current_acceptance = np.mean(self.optimizer.acceptance_window) if self.optimizer.acceptance_window else 0
            
            # Get smooth temperature-based colors
            temp_color = self.get_smooth_colors(temp_ratio)
            
            # Update main visualization with smooth trails
            if len(self.optimizer.solution_trail) > 1:
                trail_x = [sol[0] for sol in self.optimizer.solution_trail]
                trail_y = [sol[1] for sol in self.optimizer.solution_trail]
                self.main_trail.set_data(trail_x, trail_y)
                self.main_trail.set_color(temp_color)
                
                # Create gradient effect for trail
                alpha_values = np.linspace(0.2, 0.8, len(trail_x))
                self.main_trail.set_alpha(0.7)
            
            if len(self.optimizer.best_trail) > 1:
                best_x = [sol[0] for sol in self.optimizer.best_trail]
                best_y = [sol[1] for sol in self.optimizer.best_trail]
                self.best_trail.set_data(best_x, best_y)
            
            # Update markers with smooth animations
            current_pos = self.optimizer.current_solution
            proposed_pos = self.optimizer.proposed_solution
            greedy_pos = self.optimizer.greedy_solution
            best_pos = self.optimizer.best_solution
            
            # Current point with acceptance feedback
            self.current_point.set_data([current_pos[0]], [current_pos[1]])
            if accepted:
                self.current_point.set_color('white')
                self.current_point.set_markeredgecolor('lime')
                self.current_point.set_markersize(16)
            else:
                self.current_point.set_color('gray')
                self.current_point.set_markeredgecolor('red')
                self.current_point.set_markersize(12)
            
            # Proposal and greedy markers with temperature coloring
            self.proposed_point.set_data([proposed_pos[0]], [proposed_pos[1]])
            self.proposed_point.set_color(temp_color)
            
            self.greedy_point.set_data([greedy_pos[0]], [greedy_pos[1]])
            
            # Best solution marker with pulsing effect
            pulse_size = 18 + 4 * np.sin(frame * 0.3)
            self.best_point.set_data([best_pos[0]], [best_pos[1]])
            self.best_point.set_markersize(pulse_size)
            
            # Reheating effect
            if self.optimizer.reheating_intensity > 0:
                self.reheating_circle.set_center((current_pos[0], current_pos[1]))
                self.reheating_circle.set_alpha(self.optimizer.reheating_intensity * 0.8)
                self.reheating_circle.set_radius(0.2 + 0.3 * self.optimizer.reheating_intensity)
            else:
                self.reheating_circle.set_alpha(0)
            
            # Update dashboard
            self.update_dashboard_smooth()
            
            # Dynamic title with status
            status = "✓ ACCEPTED" if accepted else "✗ REJECTED"
            title_color = "#00ff00" if accepted else "#ff4444"
            
            title = (f"Hybrid MCMC-Greedy-SA Optimizer | "
                    f"Step {self.optimizer.iteration} | "
                    f"T={self.optimizer.current_temp:.4f} | "
                    f"{status}")
            
            if self.optimizer.last_reheating:
                title += " | 🔥 REHEATING!"
            
            self.fig.suptitle(title, fontsize=16, fontweight='bold', color='white')
            
            return (self.main_trail, self.best_trail, self.current_point, 
                   self.proposed_point, self.greedy_point, self.best_point)
        
        # Create smooth animation
        anim = FuncAnimation(self.fig, animate, frames=self.optimizer.max_iterations,
                           interval=80, blit=False, repeat=False)
        
        return self.fig, anim
    
    def update_dashboard_smooth(self):
        """Update dashboard with smooth transitions"""
        opt = self.optimizer
        
        # Calculate metrics
        temp_ratio = opt.current_temp / opt.initial_temp
        current_acceptance = np.mean(opt.acceptance_window) if opt.acceptance_window else 0
        
        if len(opt.best_history) > 1:
            recent_improvement = opt.best_history[-min(20, len(opt.best_history))]
            improvement_rate = (max(recent_improvement) - min(recent_improvement)) / len(recent_improvement)
        else:
            improvement_rate = 0
        
        # Update statistics
        self.iter_text.set_text(f"Iteration: {opt.iteration:,}")
        self.best_text.set_text(f"Best: {opt.best_value:.8f}")
        self.current_text.set_text(f"Current: {opt.current_value:.6f}")
        self.improvement_text.set_text(f"Improvement Rate: {improvement_rate:.2e}")
        self.cache_text.set_text(f"Cache: {len(opt.solution_cache)} evals")
        self.tabu_text.set_text(f"Tabu: {len(opt.tabu_list)}/{opt.tabu_size}")
        
        # Temperature visualization
        temp_color = self.get_smooth_colors(temp_ratio)
        self.temp_bar[0].set_width(temp_ratio)
        self.temp_bar[0].set_color(temp_color)
        self.temp_value_text.set_text(f'{opt.current_temp:.5f}')
        
        # Acceptance visualization
        accept_color = self.get_acceptance_color(current_acceptance)
        self.accept_bar[0].set_width(current_acceptance)
        self.accept_bar[0].set_color(accept_color)
        self.accept_value_text.set_text(f'{current_acceptance:.3f}')
        
        # Phase determination with smooth transitions
        if temp_ratio > 0.7:
            phase = "EXPLORATION"
            phase_color = '#ff4444'
            detail = "High temp, global search"
        elif temp_ratio > 0.3:
            phase = "TRANSITION"
            phase_color = '#ffaa00'
            detail = "Medium temp, balanced search"
        else:
            phase = "EXPLOITATION"
            phase_color = '#44aaff'
            detail = "Low temp, local refinement"
        
        self.phase_main_text.set_text(phase)
        self.phase_main_text.set_color(phase_color)
        self.phase_detail_text.set_text(detail)
        
        # Reheating status
        if opt.last_reheating:
            self.reheating_text.set_text("🔥 REHEATING ACTIVE! 🔥")
            self.reheating_text.set_color('#ff0000')
        elif len(opt.reheating_events) > 0:
            last_reheat = opt.iteration - opt.reheating_events[-1]
            self.reheating_text.set_text(f"Last reheat: {last_reheat} steps ago")
            self.reheating_text.set_color('#ff8800')
        else:
            self.reheating_text.set_text("No reheating events")
            self.reheating_text.set_color('#666666')
        
        # Convergence plot with log scale
        if len(opt.best_history) > 1:
            iterations = range(len(opt.best_history))
            self.progress_line.set_data(iterations, opt.best_history)
            self.ax_progress.relim()
            self.ax_progress.autoscale_view()
            
            # Mark reheating events with vertical lines
            for reheat_iter in opt.reheating_events:
                if reheat_iter < len(opt.best_history):
                    self.ax_progress.axvline(x=reheat_iter, color='red', 
                                           linestyle='--', alpha=0.8, linewidth=2)
    
    def get_smooth_colors(self, temp_ratio):
        """Smooth color transition for temperature"""
        # Smooth HSV-based color transition
        hue = 0.65 - 0.65 * temp_ratio  # Blue to red
        sat = 0.9
        val = 0.8 + 0.2 * temp_ratio
        return colorsys.hsv_to_rgb(hue, sat, val)
    
    def get_acceptance_color(self, acceptance_rate):
        """Color for acceptance rate visualization"""
        # Smooth transition from red (low) to green (high)
        if acceptance_rate < 0.2:
            return (1.0, 0.2, 0.2)  # Red
        elif acceptance_rate < 0.5:
            # Red to yellow transition
            t = (acceptance_rate - 0.2) / 0.3
            return (1.0, 0.2 + 0.8 * t, 0.2)
        elif acceptance_rate < 0.8:
            # Yellow to green transition
            t = (acceptance_rate - 0.5) / 0.3
            return (1.0 - 0.5 * t, 1.0, 0.2)
        else:
            # High acceptance: bright green
            return (0.2, 1.0, 0.2)

def run_smooth_hybrid_demo():
    """Run the polished hybrid optimization demonstration"""
    
    print("🎯 HYBRID MCMC-GREEDY-SA OPTIMIZER")
    print("=" * 50)
    print("✨ Enhanced Features:")
    print("  • Smooth MCMC proposals with adaptive step size")
    print("  • Intelligent greedy local search")
    print("  • Temperature-dependent SA acceptance")
    print("  • Tabu memory with distance-based avoidance")
    print("  • Smart reheating on stagnation detection")
    print("  • Efficient solution caching")
    print("  • Real-time convergence monitoring")
    print()
    print("🎲 Test Function: Rastrigin (multimodal)")
    print("🎯 Global Optimum: f(0,0) = 0")
    print("📊 Watch the smooth temperature-color transitions!")
    print()
    
    # Create optimizer with refined parameters
    optimizer = HybridMCMCGreedySAOptimizer(
        bounds=(-4, 4),
        initial_temp=3.0,        # Higher for better exploration
        cooling_rate=0.997,      # Slower cooling for smoother transitions
        min_temp=0.005,         # Lower minimum for fine-tuning
        max_iterations=800,
        tabu_size=15,
        reheating_threshold=40,
        mcmc_step_size=0.4
    )
    
    # Create smooth visualizer
    visualizer = SmoothHybridVisualizer(optimizer)
    
    # Generate animation
    print("🚀 Starting optimization animation...")
    fig, animation = visualizer.create_animation()
    
    plt.tight_layout()
    plt.show()
    
    # Comprehensive final analysis
    print("\n" + "🏁 OPTIMIZATION COMPLETE" + " " + "🏁")
    print("=" * 50)
    
    # Solution quality analysis
    distance_from_optimum = np.linalg.norm(optimizer.best_solution)
    relative_error = optimizer.best_value / max(1e-10, abs(optimizer.current_value))
    
    print(f"📍 Best Solution: [{optimizer.best_solution[0]:.6f}, {optimizer.best_solution[1]:.6f}]")
    print(f"🎯 Best Value: {optimizer.best_value:.10f}")
    print(f"📏 Distance from (0,0): {distance_from_optimum:.6f}")
    print(f"🔍 Function Evaluations: {len(optimizer.solution_cache):,}")
    print(f"🌡️  Final Temperature: {optimizer.current_temp:.6f}")
    print(f"✅ Final Acceptance Rate: {optimizer.acceptance_history[-1]:.4f}")
    
    # Performance metrics
    if optimizer.iteration > 0:
        eval_efficiency = len(optimizer.solution_cache) / (optimizer.iteration * 13)  # 13 = max evals per iteration
        print(f"💾 Cache Efficiency: {(1-eval_efficiency)*100:.1f}% savings")
        
        if len(optimizer.best_history) > 50:
            initial_best = optimizer.best_history[0]
            final_best = optimizer.best_history[-1]
            total_improvement = initial_best - final_best
            print(f"📈 Total Improvement: {total_improvement:.8f}")
            print(f"⚡ Convergence Rate: {total_improvement/optimizer.iteration:.2e} per iteration")
    
    # Reheating analysis
    if optimizer.reheating_events:
        print(f"🔥 Reheating Events: {len(optimizer.reheating_events)}")
        print(f"   Triggered at iterations: {optimizer.reheating_events}")
        
        # Calculate reheating effectiveness
        if len(optimizer.reheating_events) >= 2:
            post_reheat_improvements = []
            for reheat_iter in optimizer.reheating_events:
                if reheat_iter + 20 < len(optimizer.best_history):
                    pre_value = optimizer.best_history[reheat_iter]
                    post_value = min(optimizer.best_history[reheat_iter:reheat_iter+20])
                    if post_value < pre_value:
                        post_reheat_improvements.append(pre_value - post_value)
            
            if post_reheat_improvements:
                avg_improvement = np.mean(post_reheat_improvements)
                print(f"   Average post-reheating improvement: {avg_improvement:.6f}")
    else:
        print("🔥 No reheating events (smooth convergence)")
    
    # Solution quality assessment
    print(f"\n🏆 SOLUTION QUALITY ASSESSMENT:")
    if optimizer.best_value < 0.001:
        print("   🌟 EXCELLENT: Nearly perfect global optimum!")
        quality = "EXCELLENT"
    elif optimizer.best_value < 0.01:
        print("   ✨ VERY GOOD: High-quality near-optimal solution")
        quality = "VERY GOOD"
    elif optimizer.best_value < 0.1:
        print("   👍 GOOD: Solid optimization result")
        quality = "GOOD"
    elif optimizer.best_value < 1.0:
        print("   📊 FAIR: Reasonable solution found")
        quality = "FAIR"
    else:
        print("   🔧 NEEDS TUNING: Consider adjusting parameters")
        quality = "NEEDS TUNING"
    
    # Algorithm behavior analysis
    print(f"\n🔬 ALGORITHM BEHAVIOR ANALYSIS:")
    if len(optimizer.acceptance_history) > 100:
        early_phase = np.mean(optimizer.acceptance_history[:100])
        middle_phase = np.mean(optimizer.acceptance_history[100:400]) if len(optimizer.acceptance_history) > 400 else early_phase
        late_phase = np.mean(optimizer.acceptance_history[-100:])
        
        print(f"   Early acceptance rate: {early_phase:.3f} (exploration)")
        print(f"   Middle acceptance rate: {middle_phase:.3f} (transition)")
        print(f"   Late acceptance rate: {late_phase:.3f} (exploitation)")
        
        if early_phase > 0.7 and late_phase < 0.3:
            print("   ✅ Proper cooling schedule: High→Low acceptance")
        elif early_phase > 0.5:
            print("   ⚡ Good exploration phase detected")
        else:
            print("   🔧 Consider higher initial temperature")
    
    # Convergence analysis
    if len(optimizer.best_history) > 50:
        convergence_point = None
        tolerance = 1e-6
        
        for i in range(50, len(optimizer.best_history)):
            window = optimizer.best_history[i-50:i]
            if max(window) - min(window) < tolerance:
                convergence_point = i
                break
        
        if convergence_point:
            print(f"   🎯 Convergence achieved at iteration {convergence_point}")
            print(f"   ⏱️  Convergence efficiency: {convergence_point/optimizer.max_iterations*100:.1f}% of max iterations")
        else:
            print(f"   ⏳ Still converging (good for complex landscapes)")
    
    return optimizer, visualizer

def create_summary_analysis(optimizer):
    """Create beautiful summary plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0a0a0a')
    
    # Style all subplots
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='white')
    
    # 1. Convergence with reheating events
    iterations = range(len(optimizer.best_history))
    ax1.plot(iterations, optimizer.best_history, 'gold', linewidth=3, alpha=0.9)
    ax1.axhline(y=0, color='lime', linestyle='--', linewidth=2, alpha=0.8, label='Global Optimum')
    
    # Mark reheating events with enhanced visibility
    for i, reheat in enumerate(optimizer.reheating_events):
        if reheat < len(optimizer.best_history):
            ax1.axvline(x=reheat, color='red', linestyle='--', alpha=0.8, linewidth=3)
            if i == 0:  # Add label only once
                ax1.axvline(x=reheat, color='red', linestyle='--', alpha=0.8, 
                           linewidth=3, label='Reheating Events')
    
    ax1.set_title('Best Value Convergence', color='white', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Iteration', color='white', fontsize=12)
    ax1.set_ylabel('Best Value (log scale)', color='white', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(facecolor='black', edgecolor='white')
    
    # 2. Temperature schedule with smooth curve
    ax2.plot(optimizer.temp_history, 'red', linewidth=3, alpha=0.9)
    ax2.fill_between(range(len(optimizer.temp_history)), 0, optimizer.temp_history, 
                     color='red', alpha=0.3)
    
    # Highlight reheating spikes
    for reheat in optimizer.reheating_events:
        if reheat < len(optimizer.temp_history):
            ax2.axvline(x=reheat, color='orange', linewidth=2, alpha=0.8)
    
    ax2.set_title('Temperature Evolution', color='white', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Iteration', color='white', fontsize=12)
    ax2.set_ylabel('Temperature', color='white', fontsize=12)
    
    # 3. Acceptance rate with moving average
    if len(optimizer.acceptance_history) > 10:
        # Smooth the acceptance rate with moving average
        window_size = min(20, len(optimizer.acceptance_history) // 4)
        smoothed_acceptance = np.convolve(optimizer.acceptance_history, 
                                        np.ones(window_size)/window_size, mode='valid')
        
        ax3.plot(optimizer.acceptance_history, 'cyan', alpha=0.5, linewidth=1, label='Raw')
        ax3.plot(range(window_size-1, len(optimizer.acceptance_history)), 
                smoothed_acceptance, 'cyan', linewidth=3, label='Smoothed')
        
        # Show ideal acceptance zones
        ax3.axhspan(0.4, 0.6, alpha=0.2, color='green', label='Ideal Range')
        
        ax3.set_title('Acceptance Rate Evolution', color='white', fontweight='bold', fontsize=14)
        ax3.set_xlabel('Iteration', color='white', fontsize=12)
        ax3.set_ylabel('Acceptance Rate', color='white', fontsize=12)
        ax3.legend(facecolor='black', edgecolor='white')
    
    # 4. Search space exploration heatmap
    if len(optimizer.solution_cache) > 20:
        # Create high-resolution exploration map
        solutions = list(optimizer.solution_cache.keys())
        x_coords = [sol[0] for sol in solutions]
        y_coords = [sol[1] for sol in solutions]
        
        # 2D histogram with higher resolution
        hist, xedges, yedges = np.histogram2d(
            x_coords, y_coords, bins=40, 
            range=[[optimizer.bounds[0], optimizer.bounds[1]], 
                   [optimizer.bounds[0], optimizer.bounds[1]]]
        )
        
        # Smooth heatmap
        from scipy.ndimage import gaussian_filter
        hist_smooth = gaussian_filter(hist, sigma=0.8)
        
        im = ax4.imshow(hist_smooth.T, origin='lower', 
                       extent=[optimizer.bounds[0], optimizer.bounds[1], 
                              optimizer.bounds[0], optimizer.bounds[1]], 
                       cmap='hot', alpha=0.9, interpolation='bilinear')
        
        # Mark key points
        ax4.plot(0, 0, '*', markersize=20, color='lime', 
                markeredgecolor='white', markeredgewidth=3, label='Global Optimum')
        ax4.plot(optimizer.best_solution[0], optimizer.best_solution[1], 
                '*', markersize=18, color='gold', markeredgecolor='red', 
                markeredgewidth=2, label='Best Found')
        
        ax4.set_title('Exploration Density', color='white', fontweight='bold', fontsize=14)
        ax4.set_xlabel('X', color='white', fontsize=12)
        ax4.set_ylabel('Y', color='white', fontsize=12)
        ax4.legend(facecolor='black', edgecolor='white')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Visit Frequency', color='white', fontsize=10)
        cbar.ax.tick_params(colors='white')
    else:
        ax4.text(0.5, 0.5, 'Insufficient Data\nfor Heatmap', 
                ha='center', va='center', transform=ax4.transAxes, 
                color='white', fontsize=14, fontweight='bold')
        ax4.set_title('Exploration Density', color='white', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    return fig

def performance_benchmark():
    """Benchmark different algorithm configurations"""
    print("\n🏃‍♂️ PERFORMANCE BENCHMARK")
    print("=" * 40)
    
    configurations = [
        {
            "name": "Aggressive Cooling",
            "params": {"initial_temp": 2.0, "cooling_rate": 0.992, "reheating_threshold": 30}
        },
        {
            "name": "Conservative Cooling", 
            "params": {"initial_temp": 2.0, "cooling_rate": 0.998, "reheating_threshold": 60}
        },
        {
            "name": "High Temperature",
            "params": {"initial_temp": 5.0, "cooling_rate": 0.997, "reheating_threshold": 40}
        },
        {
            "name": "No Reheating",
            "params": {"initial_temp": 3.0, "cooling_rate": 0.997, "reheating_threshold": 9999}
        },
        {
            "name": "Optimal (Default)",
            "params": {"initial_temp": 3.0, "cooling_rate": 0.997, "reheating_threshold": 40}
        }
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing {config['name']}...", end=" ")
        
        # Run multiple trials
        trial_results = []
        for trial in range(3):
            optimizer = HybridMCMCGreedySAOptimizer(
                bounds=(-4, 4),
                max_iterations=400,
                **config['params']
            )
            
            # Run optimization silently
            for _ in range(400):
                optimizer.step()
                if optimizer.is_converged():
                    break
            
            trial_results.append({
                'best_value': optimizer.best_value,
                'iterations_to_converge': optimizer.iteration,
                'reheating_count': len(optimizer.reheating_events),
                'function_evals': len(optimizer.solution_cache)
            })
        
        # Average results
        avg_best = np.mean([r['best_value'] for r in trial_results])
        avg_iters = np.mean([r['iterations_to_converge'] for r in trial_results])
        avg_reheats = np.mean([r['reheating_count'] for r in trial_results])
        avg_evals = np.mean([r['function_evals'] for r in trial_results])
        
        results.append({
            'name': config['name'],
            'avg_best': avg_best,
            'avg_iterations': avg_iters,
            'avg_reheats': avg_reheats,
            'avg_evals': avg_evals
        })
        
        print(f"Best: {avg_best:.6f}")
    
    # Display benchmark results
    print(f"\n📊 BENCHMARK RESULTS (averaged over 3 trials):")
    print("-" * 70)
    print(f"{'Configuration':<20} {'Best Value':<12} {'Iterations':<12} {'Reheats':<10} {'Evals':<8}")
    print("-" * 70)
    
    for result in sorted(results, key=lambda x: x['avg_best']):
        print(f"{result['name']:<20} {result['avg_best']:<12.6f} "
              f"{result['avg_iterations']:<12.1f} {result['avg_reheats']:<10.1f} "
              f"{result['avg_evals']:<8.0f}")
    
    # Find best configuration
    best_config = min(results, key=lambda x: x['avg_best'])
    print(f"\n🏆 BEST CONFIGURATION: {best_config['name']}")
    print(f"   Achieved average best value: {best_config['avg_best']:.8f}")
    
    return results

def detailed_landscape_analysis():
    """Analyze the optimization landscape in detail"""
    print("\n🗺️  DETAILED LANDSCAPE ANALYSIS")
    print("=" * 40)
    
    # Create high-resolution landscape
    bounds = (-4, 4)
    resolution = 200
    x_range = np.linspace(bounds[0], bounds[1], resolution)
    y_range = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    optimizer = HybridMCMCGreedySAOptimizer()
    Z = np.zeros_like(X)
    
    print("Calculating landscape (high resolution)...", end=" ")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = optimizer.rastrigin_function([X[i, j], Y[i, j]])
    print("Done!")
    
    # Find approximate local minima
    local_minima = []
    step = 8
    threshold = 0.1
    
    for i in range(step, X.shape[0]-step, step):
        for j in range(step, X.shape[1]-step, step):
            local_patch = Z[i-step:i+step+1, j-step:j+step+1]
            if Z[i, j] == np.min(local_patch) and Z[i, j] < threshold:
                local_minima.append((X[i, j], Y[i, j], Z[i, j]))
    
    # Remove duplicates and sort
    unique_minima = []
    for candidate in local_minima:
        is_unique = True
        for existing in unique_minima:
            if np.linalg.norm([candidate[0] - existing[0], candidate[1] - existing[1]]) < 0.3:
                is_unique = False
                break
        if is_unique:
            unique_minima.append(candidate)
    
    unique_minima.sort(key=lambda x: x[2])
    
    print(f"📍 Local minima found: {len(unique_minima)}")
    print(f"🎯 Global minimum: f(0,0) = 0.000000")
    print(f"🌍 Search domain: [{bounds[0]}, {bounds[1]}]²")
    print(f"📏 Domain area: {(bounds[1] - bounds[0])**2}")
    
    if unique_minima:
        print(f"\n🏔️  TOP LOCAL MINIMA:")
        for i, (x, y, val) in enumerate(unique_minima[:8]):
            distance_from_global = np.sqrt(x**2 + y**2)
            print(f"   {i+1:2d}. f({x:+6.3f}, {y:+6.3f}) = {val:8.6f} "
                  f"[dist={distance_from_global:5.3f}]")
    
    # Calculate landscape statistics
    gradient_magnitude = np.sqrt(np.gradient(Z)[0]**2 + np.gradient(Z)[1]**2)
    avg_gradient = np.mean(gradient_magnitude)
    max_gradient = np.max(gradient_magnitude)
    
    print(f"\n📈 LANDSCAPE DIFFICULTY METRICS:")
    print(f"   Average gradient magnitude: {avg_gradient:.4f}")
    print(f"   Maximum gradient magnitude: {max_gradient:.4f}")
    print(f"   Function value range: [{np.min(Z):.6f}, {np.max(Z):.2f}]")
    print(f"   Estimated local minima density: {len(unique_minima)/(bounds[1]-bounds[0])**2:.2f} per unit²")
    
    return X, Y, Z, unique_minima

# Main execution with smooth flow
if __name__ == "__main__":
    print("🚀 HYBRID MCMC-GREEDY-SA OPTIMIZATION SUITE")
    print("=" * 60)
    print("🎨 Smooth Animation & Comprehensive Analysis")
    print("=" * 60)
    
    # 1. Landscape analysis
    X, Y, Z, minima = detailed_landscape_analysis()
    
    print(f"\n⏳ Preparing smooth visualization...")
    
    # 2. Main optimization with smooth animation
    optimizer, visualizer = run_smooth_hybrid_demo()
    
    # 3. Create summary analysis
    print(f"\n📊 Generating summary analysis...")
    summary_fig = create_summary_analysis(optimizer)
    
    # 4. Performance benchmark
    benchmark_results = performance_benchmark()
    
    # 5. Final comprehensive report
    print(f"\n" + "🎉 ANALYSIS COMPLETE!" + " " + "🎉")
    print("=" * 60)
    
    # Calculate final metrics
    solution_quality = "EXCELLENT" if optimizer.best_value < 0.001 else \
                      "VERY GOOD" if optimizer.best_value < 0.01 else \
                      "GOOD" if optimizer.best_value < 0.1 else "FAIR"
    
    efficiency_score = len(optimizer.solution_cache) / (optimizer.iteration * 13)
    efficiency_rating = "EXCELLENT" if efficiency_score < 0.7 else \
                       "GOOD" if efficiency_score < 0.8 else \
                       "FAIR" if efficiency_score < 0.9 else "POOR"
    
    print(f"🏆 FINAL PERFORMANCE SUMMARY:")
    print(f"   Solution Quality: {solution_quality}")
    print(f"   Cache Efficiency: {efficiency_rating} ({(1-efficiency_score)*100:.1f}% saved)")
    print(f"   Convergence: {len(optimizer.best_history)} iterations")
    print(f"   Reheating Events: {len(optimizer.reheating_events)}")
    print(f"   Distance to Global Optimum: {np.linalg.norm(optimizer.best_solution):.6f}")
    
    print(f"\n✨ Features Successfully Demonstrated:")
    print(f"   ✅ Smooth MCMC proposals with temperature adaptation")
    print(f"   ✅ Intelligent greedy local refinement")
    print(f"   ✅ Metropolis acceptance with proper cooling")
    print(f"   ✅ Tabu memory preventing cycles")
    print(f"   ✅ Adaptive reheating on stagnation")
    print(f"   ✅ Efficient solution caching")
    print(f"   ✅ Real-time convergence monitoring")
    print(f"   ✅ Beautiful smooth animations")
    
    plt.show()  # Show summary plot
    #return optimizer, visualizer, summary_fig, benchmark_results