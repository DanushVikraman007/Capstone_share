# YO (Yukti Opis): Hybrid MCMC + Greedy + SA + Tabu Optimization Framework
# Complete implementation for environmental data matching

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUTO-TUNED HYPERPARAMETERS
# ============================================================================

def auto_tune_hyperparameters(N):
    """Auto-tune hyperparameters based on dataset size N"""
    hyperparams = {
        'chains': max(2, min(8, N // 500)),
        'neighborhood_size': min(256, max(32, N // 20)),
        'prefilter_size': lambda ns: min(ns // 4, N),
        'initial_T': 1.0,
        'cooling_rate': 0.995 if N > 1000 else 0.99,
        'tabu_size': max(50, N // 50),
        'reheating_window': max(100, N // 20),
        'reheating_alpha': 0.02,
        'reheating_multiplier': 2.0,
        'reheating_temp_ratio': 0.5,
        'max_iters': min(2000, max(500, N // 2)),
        'diversity_weight': 0.10,
        'target_acceptance': 0.5
    }
    
    # Calculate prefilter_size based on neighborhood_size
    hyperparams['prefilter_size'] = min(hyperparams['neighborhood_size'] // 4, N)
    
    return hyperparams

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data(csv_file):
    """Load CSV and validate required columns"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns (case-insensitive)
        required_cols = ['temperature', 'humidity', 'pressure']
        column_mapping = {}
        
        for req_col in required_cols:
            found = False
            for col in df.columns:
                if req_col.lower() in col.lower():
                    column_mapping[req_col] = col
                    found = True
                    break
            if not found:
                raise ValueError(f"Required column '{req_col}' not found in CSV")
        
        print(f"Column mapping: {column_mapping}")
        
        # Clean data
        original_len = len(df)
        df_clean = df.dropna(subset=list(column_mapping.values()))
        dropped = original_len - len(df_clean)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values")
        
        return df_clean, column_mapping
        
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return None, None

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_weighted_normalized_error(row, column_mapping, target_conditions):
    """Calculate weighted normalized error for scoring"""
    errors = []
    weights = {'temperature': 1.0, 'humidity': 1.0, 'pressure': 1.0}  # Equal weights
    
    for feature, target_value in target_conditions.items():
        actual_col = column_mapping[feature]
        actual_value = row[actual_col]
        
        if pd.isna(actual_value):
            errors.append(1.0)  # Maximum error for missing values
            continue
        
        # Normalize error based on feature scale
        if feature == 'temperature':
            scale = 50.0  # Assume temperature range ~50°C
        elif feature == 'humidity':
            scale = 1.0   # Humidity is 0-1
        elif feature == 'pressure':
            scale = 100.0 # Assume pressure range ~100 hPa
        else:
            scale = 1.0
        
        normalized_error = abs(actual_value - target_value) / scale
        errors.append(normalized_error * weights[feature])
    
    weighted_error = np.mean(errors)
    accuracy = 1.0 / (1.0 + weighted_error)  # Normalized inverse error
    
    return accuracy, weighted_error

def find_closest_match(df, column_mapping, target_conditions):
    """Find the actual closest match in the dataset (ground truth)"""
    best_accuracy = 0.0
    best_idx = 0
    
    for idx, row in df.iterrows():
        accuracy, _ = calculate_weighted_normalized_error(row, column_mapping, target_conditions)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_idx = idx
    
    return best_idx, best_accuracy

# ============================================================================
# YO CHAIN IMPLEMENTATION
# ============================================================================

class YOChain:
    """Single YO (Yukti Opis) optimization chain"""
    
    def __init__(self, df, column_mapping, target_conditions, hyperparams, chain_id=0):
        self.df = df
        self.column_mapping = column_mapping
        self.target_conditions = target_conditions
        self.params = hyperparams
        self.chain_id = chain_id
        
        # Chain state
        self.current_idx = np.random.randint(0, len(df))
        self.current_accuracy = 0.0
        self.temperature = hyperparams['initial_T']
        self.iteration = 0
        
        # Tracking
        self.accuracy_history = []
        self.best_accuracy_history = []
        self.position_history = []  # For exploration path visualization
        self.acceptance_history = []
        self.temperature_history = []
        self.reheating_events = []
        
        # Tabu memory
        self.tabu_list = []
        
        # Reheating tracking
        self.last_improvement = 0
        self.best_accuracy = 0.0
        self.best_idx = self.current_idx
        
        # Performance metrics
        self.total_acceptances = 0
        self.total_proposals = 0
        
        # Initialize current state
        self._update_current_state()
    
    def _update_current_state(self):
        """Update current state accuracy and tracking"""
        row = self.df.iloc[self.current_idx]
        self.current_accuracy, _ = calculate_weighted_normalized_error(
            row, self.column_mapping, self.target_conditions
        )
        
        # Update best if improved
        if self.current_accuracy > self.best_accuracy:
            self.best_accuracy = self.current_accuracy
            self.best_idx = self.current_idx
            self.last_improvement = self.iteration
    
    def get_neighborhood(self, size):
        """Get random neighborhood of candidates"""
        n_candidates = min(size, len(self.df))
        return np.random.choice(len(self.df), size=n_candidates, replace=False)
    
    def greedy_local_refinement(self):
        """Greedy local search after accepted moves"""
        neighborhood = self.get_neighborhood(self.params['neighborhood_size'] // 4)  # Smaller for refinement
        best_candidate = self.current_idx
        best_accuracy = self.current_accuracy
        
        for candidate_idx in neighborhood:
            if candidate_idx in [tabu[0] for tabu in self.tabu_list[-10:]]:  # Check recent tabu
                continue
            
            row = self.df.iloc[candidate_idx]
            accuracy, _ = calculate_weighted_normalized_error(
                row, self.column_mapping, self.target_conditions
            )
            
            if accuracy > best_accuracy:
                best_candidate = candidate_idx
                best_accuracy = accuracy
        
        return best_candidate, best_accuracy
    
    def mcmc_step_with_large_jumps(self):
        """MCMC step with occasional large jumps for global exploration"""
        self.total_proposals += 1
        
        # Decide on jump type
        if np.random.random() < 0.1:  # 10% large jumps
            # Large jump: sample from entire dataset
            candidates = self.get_neighborhood(self.params['neighborhood_size'])
        else:
            # Local jump: smaller neighborhood
            local_size = min(self.params['neighborhood_size'] // 2, 64)
            candidates = self.get_neighborhood(local_size)
        
        # Find best candidate in neighborhood
        best_candidate = None
        best_accuracy = -1
        
        for candidate_idx in candidates:
            # Apply tabu penalty
            tabu_penalty = 0.0
            for tabu_idx, tabu_iter in self.tabu_list:
                if tabu_idx == candidate_idx:
                    age = self.iteration - tabu_iter
                    tabu_penalty += 0.1 * np.exp(-age / 50)
            
            row = self.df.iloc[candidate_idx]
            accuracy, _ = calculate_weighted_normalized_error(
                row, self.column_mapping, self.target_conditions
            )
            
            # Apply tabu penalty
            accuracy = max(0, accuracy - tabu_penalty)
            
            if accuracy > best_accuracy:
                best_candidate = candidate_idx
                best_accuracy = accuracy
        
        if best_candidate is None:
            return False
        
        # Simulated Annealing acceptance decision
        accuracy_diff = best_accuracy - self.current_accuracy
        if accuracy_diff > 0 or np.random.random() < np.exp(accuracy_diff / self.temperature):
            # Accept move
            self.current_idx = best_candidate
            self.current_accuracy = best_accuracy
            
            # Update tabu list
            self.tabu_list.append((best_candidate, self.iteration))
            if len(self.tabu_list) > self.params['tabu_size']:
                self.tabu_list.pop(0)
            
            self.total_acceptances += 1
            
            # Greedy refinement after acceptance
            refined_idx, refined_accuracy = self.greedy_local_refinement()
            if refined_accuracy > self.current_accuracy:
                self.current_idx = refined_idx
                self.current_accuracy = refined_accuracy
            
            self._update_current_state()
            return True
        
        return False
    
    def update_temperature(self):
        """Update temperature with geometric cooling and reheating"""
        # Check for reheating
        if (self.iteration - self.last_improvement) >= self.params['reheating_window']:
            if np.random.random() < self.params['reheating_alpha']:
                old_temp = self.temperature
                self.temperature = self.params['initial_T'] * self.params['reheating_temp_ratio']
                self.reheating_events.append(self.iteration)
                self.last_improvement = self.iteration
                print(f"Chain {self.chain_id}: Reheating at iteration {self.iteration} "
                      f"(T: {old_temp:.6f} -> {self.temperature:.6f})")
                return
        
        # Regular geometric cooling
        self.temperature *= self.params['cooling_rate']
    
    def run_chain(self):
        """Run the complete YO optimization chain"""
        print(f"Starting YO chain {self.chain_id}")
        
        for i in range(self.params['max_iters']):
            self.iteration = i
            
            # MCMC step with occasional large jumps
            accepted = self.mcmc_step_with_large_jumps()
            
            # Record history
            self.accuracy_history.append(self.current_accuracy)
            self.best_accuracy_history.append(self.best_accuracy)
            self.acceptance_history.append(1 if accepted else 0)
            self.temperature_history.append(self.temperature)
            
            # Store position for exploration path (simplified: just store index)
            self.position_history.append(self.current_idx)
            
            # Update temperature
            self.update_temperature()
            
            # Progress reporting
            if i % 500 == 0 or i == self.params['max_iters'] - 1:
                acceptance_rate = self.total_acceptances / max(1, self.total_proposals)
                print(f"Chain {self.chain_id}: Iteration {i}, "
                      f"Best Accuracy: {self.best_accuracy:.6f}, "
                      f"Current: {self.current_accuracy:.6f}, "
                      f"Acceptance Rate: {acceptance_rate:.3f}, "
                      f"Temperature: {self.temperature:.6f}")
        
        final_acceptance_rate = self.total_acceptances / max(1, self.total_proposals)
        print(f"Chain {self.chain_id} completed: Best accuracy: {self.best_accuracy:.6f}, "
              f"Final acceptance rate: {final_acceptance_rate:.3f}, "
              f"Reheating events: {len(self.reheating_events)}")

# ============================================================================
# MAIN YO FRAMEWORK
# ============================================================================

def run_yo_optimization(df, column_mapping, target_conditions, hyperparams):
    """Run complete YO optimization with multiple chains"""
    print("=" * 80)
    print("YO (Yukti Opis) Hybrid Optimization Framework")
    print("=" * 80)
    
    start_time = time.time()
    
    print("Starting YO optimization chains...")
    chains = []
    
    # Run multiple chains in parallel (simulated)
    for chain_id in range(hyperparams['chains']):
        chain = YOChain(df, column_mapping, target_conditions, hyperparams, chain_id)
        chain.run_chain()
        chains.append(chain)
    
    # Find best result across all chains
    best_chain = max(chains, key=lambda c: c.best_accuracy)
    best_idx = best_chain.best_idx
    best_accuracy = best_chain.best_accuracy
    
    execution_time = time.time() - start_time
    
    # Performance metrics
    total_acceptances = sum(chain.total_acceptances for chain in chains)
    total_proposals = sum(chain.total_proposals for chain in chains)
    overall_acceptance_rate = total_acceptances / max(1, total_proposals)
    total_reheating_events = sum(len(chain.reheating_events) for chain in chains)
    
    print(f"\nYO Optimization completed in {execution_time:.2f} seconds")
    print(f"Best match found: Index {best_idx}, Accuracy: {best_accuracy:.6f}")
    print(f"Overall acceptance rate: {overall_acceptance_rate:.3f}")
    print(f"Total reheating events: {total_reheating_events}")
    
    return best_idx, best_accuracy, chains, execution_time, {
        'acceptance_rate': overall_acceptance_rate,
        'reheating_events': total_reheating_events,
        'execution_time': execution_time
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_exploration_paths(chains, df, column_mapping):
    """Plot Markov chain exploration paths"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YO Chain Exploration Paths', fontsize=16)
    
    for i, chain in enumerate(chains[:4]):  # Plot up to 4 chains
        ax = axes[i//2, i%2]
        
        # Plot accuracy over time
        iterations = range(len(chain.best_accuracy_history))
        ax.plot(iterations, chain.best_accuracy_history, 'b-', alpha=0.8, linewidth=2, label='Best Accuracy')
        ax.plot(iterations, chain.accuracy_history, 'g-', alpha=0.4, label='Current Accuracy')
        
        # Mark reheating events
        for reheat_iter in chain.reheating_events:
            ax.axvline(x=reheat_iter, color='red', linestyle='--', alpha=0.7, label='Reheating' if reheat_iter == chain.reheating_events[0] else "")
        
        ax.set_title(f'Chain {i} (Final Accuracy: {chain.best_accuracy:.4f})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yo_exploration_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_yo_vs_actual_hit(df, column_mapping, target_conditions, yo_idx, yo_accuracy, actual_idx, actual_accuracy):
    """Plot YO's best hit vs actual closest hit"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    temp_col = column_mapping['temperature']
    humid_col = column_mapping['humidity']
    press_col = column_mapping['pressure']
    
    # Sample data for visualization if too large
    if len(df) > 5000:
        sample_df = df.sample(5000)
        print("Downsampled data for visualization")
    else:
        sample_df = df
    
    pairs = [
        (temp_col, humid_col, 'Temperature vs Humidity', target_conditions['temperature'], target_conditions['humidity']),
        (temp_col, press_col, 'Temperature vs Pressure', target_conditions['temperature'], target_conditions['pressure']),
        (humid_col, press_col, 'Humidity vs Pressure', target_conditions['humidity'], target_conditions['pressure'])
    ]
    
    for i, (x_col, y_col, title, target_x, target_y) in enumerate(pairs):
        ax = axes[i]
        
        # Plot all points
        ax.scatter(sample_df[x_col], sample_df[y_col], alpha=0.3, s=8, c='lightblue', label='All data')
        
        # Plot target
        ax.scatter(target_x, target_y, s=200, c='red', marker='.', 
                  label=f'Target', edgecolors='black', linewidth=2)
        
        # Plot YO's best hit
        yo_row = df.iloc[yo_idx]
        ax.scatter(yo_row[x_col], yo_row[y_col], s=100, c='green', marker='^', 
                  label=f'YO Best (Acc: {yo_accuracy:.4f})', edgecolors='black', linewidth=1)
        
        # Plot actual closest hit
        actual_row = df.iloc[actual_idx]
        ax.scatter(actual_row[x_col], actual_row[y_col], s=100, c='orange', marker='.', 
                  label=f'Actual Closest (Acc: {actual_accuracy:.4f})', edgecolors='black', linewidth=1)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yo_vs_actual_hit.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_over_time(chains):
    """Plot time vs accuracy performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YO Performance Analysis', fontsize=16)
    
    # 1. Best accuracy convergence
    ax1 = axes[0, 0]
    for i, chain in enumerate(chains):
        iterations = range(len(chain.best_accuracy_history))
        ax1.plot(iterations, chain.best_accuracy_history, alpha=0.7, label=f'Chain {i}')
    
    ax1.set_title('Best Accuracy Convergence')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Best Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature schedules
    ax2 = axes[0, 1]
    for i, chain in enumerate(chains):
        iterations = range(len(chain.temperature_history))
        ax2.plot(iterations, chain.temperature_history, alpha=0.7, label=f'Chain {i}')
    
    ax2.set_title('Temperature Schedules')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Temperature')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Acceptance rates
    ax3 = axes[1, 0]
    for i, chain in enumerate(chains):
        # Calculate rolling acceptance rate
        window = 100
        acceptance_rate = pd.Series(chain.acceptance_history).rolling(window, min_periods=1).mean()
        iterations = range(len(acceptance_rate))
        ax3.plot(iterations, acceptance_rate, alpha=0.7, label=f'Chain {i}')
    
    ax3.set_title('Acceptance Rate Over Time')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Acceptance Rate')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Reheating events histogram
    ax4 = axes[1, 1]
    all_reheating_times = []
    for chain in chains:
        all_reheating_times.extend(chain.reheating_events)
    
    if all_reheating_times:
        ax4.hist(all_reheating_times, bins=20, alpha=0.7, color='orange')
        ax4.set_title('Reheating Events Distribution')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Frequency')
    else:
        ax4.text(0.5, 0.5, 'No Reheating Events', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Reheating Events Distribution')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('yo_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def yo_main(csv_file, target_conditions):
    """
    Main YO optimization function
    
    Args:
        csv_file (str): Path to CSV file
        target_conditions (dict): Target conditions, e.g., 
                                 {'temperature': 25.0, 'humidity': 0.6, 'pressure': 1013.0}
    
    Returns:
        dict: Results including best match, accuracy, and performance metrics
    """
    
    print("YO (Yukti Opis) Hybrid Optimization Framework")
    print("=" * 80)
    
    # Load and validate data
    df, column_mapping = load_and_validate_data(csv_file)
    if df is None:
        return None
    
    N = len(df)
    print(f"\nDataset size: {N:,} points")
    
    # Auto-tune hyperparameters
    hyperparams = auto_tune_hyperparameters(N)
    
    print("\nAuto-tuned hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    print(f"\nTarget conditions: {target_conditions}")
    print(f"Column mapping: {column_mapping}")
    
    # Find actual closest match (ground truth)
    print("\nFinding actual closest match...")
    actual_idx, actual_accuracy = find_closest_match(df, column_mapping, target_conditions)
    print(f"Actual closest match: Index {actual_idx}, Accuracy: {actual_accuracy:.6f}")
    
    # Run YO optimization
    yo_idx, yo_accuracy, chains, exec_time, metrics = run_yo_optimization(
        df, column_mapping, target_conditions, hyperparams
    )
    
    # Print diagnostic summary
    print("\n" + "=" * 80)
    print("YO OPTIMIZATION DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Dataset size: {N:,}")
    print(f"Chains: {hyperparams['chains']}")
    print(f"Max iterations per chain: {hyperparams['max_iters']:,}")
    print(f"Total iterations: {hyperparams['chains'] * hyperparams['max_iters']:,}")
    print(f"Execution time: {exec_time:.2f} seconds")
    print(f"Overall acceptance rate: {metrics['acceptance_rate']:.3f}")
    print(f"Total reheating events: {metrics['reheating_events']}")
    
    print(f"\nResults:")
    print(f"YO Best Match: Index {yo_idx}, Accuracy: {yo_accuracy:.6f}")
    print(f"Actual Closest: Index {actual_idx}, Accuracy: {actual_accuracy:.6f}")
    print(f"YO Performance: {(yo_accuracy/actual_accuracy)*100:.2f}% of optimal")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_exploration_paths(chains, df, column_mapping)
    plot_yo_vs_actual_hit(df, column_mapping, target_conditions, 
                         yo_idx, yo_accuracy, actual_idx, actual_accuracy)
    plot_performance_over_time(chains)
    
    # Prepare results
    results = {
        'yo_best_index': yo_idx,
        'yo_best_accuracy': yo_accuracy,
        'actual_best_index': actual_idx,
        'actual_best_accuracy': actual_accuracy,
        'performance_ratio': yo_accuracy / actual_accuracy,
        'execution_time': exec_time,
        'hyperparameters': hyperparams,
        'metrics': metrics,
        'chains': chains,
        'yo_best_row': df.iloc[yo_idx].to_dict(),
        'actual_best_row': df.iloc[actual_idx].to_dict()
    }
    
    print("\nYO optimization completed successfully!")
    print("Generated files:")
    print("  - yo_exploration_paths.png")
    print("  - yo_vs_actual_hit.png") 
    print("  - yo_performance_analysis.png")
    
    return results

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("YO (Yukti Opis) Hybrid Optimization Framework")
    print("=" * 80)
    print("Example usage:")
    print()
    print("# Define target conditions")
    print("target = {")
    print("    'temperature': 25.0,  # °C")
    print("    'humidity': 0.6,      # decimal (0.0-1.0)")
    print("    'pressure': 1013.0    # hPa")
    print("}")
    print()
    print("# Run optimization")
    print("results = yo_main('your_data.csv', target)")
    print()
    print("# Access results")
    print("print(f'Best match accuracy: {results[\"yo_best_accuracy\"]:.6f}')")
    print("print(f'Performance: {results[\"performance_ratio\"]*100:.2f}% of optimal')")
    print()
    print("Ready to run! Provide CSV file path and target conditions.")
    
    # Uncomment below to run with sample data
target_conditions = {
     'temperature': 25.0,
     'humidity': 0.6,
     'pressure': 1013.0
 }
results = yo_main('DATA.csv', target_conditions)