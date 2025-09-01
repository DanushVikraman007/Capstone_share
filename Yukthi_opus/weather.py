# yukthi_opusMethod: MCMC + Greedy + Simulated Annealing for Datapoint Prioritization
# Complete implementation ready for DATA.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION SECTION - MODIFY THESE VALUES AS NEEDED
# ============================================================================

# INPUT PARAMETERS (Set these before running)
CSV_FILE = "DATA.csv"  # Your uploaded CSV file
RANGES = {
    'temperature': [10, 15],  # [min, max] - MODIFY THESE
    'humidity': [0.4, 0.5],     # [min, max] - MODIFY THESE  
    'pressure': [1015, 1016]  # [min, max] - MODIFY THESE
}

# yukthi_opusPERPARAMETERS (Default values - can be modified)
HYPERPARAMS = {
    'chains': 4,
    'neighborhood_size': 128,
    'prefilter_size': 32,  # Used if neighborhood_size > dataset size
    'initial_T': 1.0,
    'cooling_rate': 0.995,
    'tabu_size': 100,
    'reheating_window': 200,
    'reheating_alpha': 0.02,
    'reheating_multiplier': 2.0,
    'reheating_temp_ratio': 0.5,
    'max_iters': 500,
    'diversity_weight': 0.10,  # λ in scoring formula
    'target_acceptance': 0.5   # Target acceptance rate for adaptive cooling
}

print("Configuration loaded. Hyperparameters:")
for key, value in HYPERPARAMS.items():
    print(f"  {key}: {value}")
print(f"\nTarget ranges: {RANGES}")

# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_and_validate_data(csv_file):
    """Load CSV and validate required columns"""
    try:
        df = pd.read_csv(r"C:\Users\user\Desktop\CAPSTONE\Vikasa_benchmark\DATA.csv")
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
        
        # Check for missing values
        missing_counts = {}
        for req_col, actual_col in column_mapping.items():
            missing = df[actual_col].isna().sum()
            missing_counts[req_col] = missing
            if missing > 0:
                print(f"Missing values in {actual_col}: {missing}")
        
        # Drop rows with missing values in required columns
        original_len = len(df)
        df_clean = df.dropna(subset=list(column_mapping.values()))
        dropped = original_len - len(df_clean)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing values")
        
        return df_clean, column_mapping
        
    except FileNotFoundError:
        print(f"ERROR: File '{csv_file}' not found. Please upload DATA.csv first.")
        return None, None
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return None, None

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_feature_score(value, range_min, range_max):
    """Calculate feature score for a single value and range"""
    if pd.isna(value):
        return 0.0
    
    if range_min <= value <= range_max:
        return 1.0
    
    # Distance-based scoring for values outside range
    span = range_max - range_min
    if span == 0:
        span = 1e-6  # Avoid division by zero
    
    if value < range_min:
        distance = range_min - value
    else:
        distance = value - range_max
    
    return max(0.0, 1.0 - distance / span)

def calculate_match_score(row, column_mapping, ranges):
    """Calculate match score for a datapoint"""
    scores = []
    for feature, (range_min, range_max) in ranges.items():
        actual_col = column_mapping[feature]
        value = row[actual_col]
        score = calculate_feature_score(value, range_min, range_max)
        scores.append(score)
    
    return np.mean(scores)

def calculate_diversity_score(candidate_idx, selected_indices, feature_data):
    """Calculate diversity score based on distance to selected points"""
    if len(selected_indices) == 0:
        return 1.0
    
    candidate_point = feature_data[candidate_idx:candidate_idx+1]
    selected_points = feature_data[selected_indices]
    
    distances = euclidean_distances(candidate_point, selected_points)
    min_distance = np.min(distances)
    
    # Normalize by approximate dataset diameter
    max_distance = np.sqrt(feature_data.shape[1])  # Rough normalization
    return min(1.0, min_distance / max_distance)

def calculate_final_score(candidate_idx, row, selected_indices, feature_data, 
                         column_mapping, ranges, diversity_weight):
    """Calculate final yukthi_opusscore"""
    match_score = calculate_match_score(row, column_mapping, ranges)
    diversity_score = calculate_diversity_score(candidate_idx, selected_indices, feature_data)
    
    final_score = match_score * (1 - diversity_weight) + diversity_score * diversity_weight
    return final_score, match_score, diversity_score

# ============================================================================
# yukthi_opusIMPLEMENTATION
# ============================================================================

class VivetaChain:
    """Single MCMC chain for yukthi_opusalgorithm"""
    
    def __init__(self, df, feature_data, column_mapping, ranges, hyperparams, chain_id=0):
        self.df = df
        self.feature_data = feature_data
        self.column_mapping = column_mapping
        self.ranges = ranges
        self.params = hyperparams
        self.chain_id = chain_id
        
        # Chain state
        self.selected_indices = []
        self.current_score = 0.0
        self.temperature = hyperparams['initial_T']
        self.iteration = 0
        
        # Tracking
        self.score_history = []
        self.best_score_history = []
        self.acceptance_history = []
        self.temperature_history = []
        self.hit_iterations = []  # When prioritized hits were found
        
        # Tabu memory
        self.tabu_list = []
        
        # Reheating
        self.last_improvement = 0
        self.best_score = 0.0
        
    def get_neighborhood(self, size):
        """Get random neighborhood of candidates"""
        n_candidates = min(size, len(self.df))
        return np.random.choice(len(self.df), size=n_candidates, replace=False)
    
    def greedy_local_improvement(self):
        """Greedy local search step"""
        neighborhood = self.get_neighborhood(self.params['neighborhood_size'])
        best_candidate = None
        best_score = self.current_score
        best_details = None
        
        for candidate_idx in neighborhood:
            if candidate_idx in self.selected_indices:
                continue
                
            # Apply tabu penalty
            tabu_penalty = 0.0
            for tabu_idx, tabu_iter in self.tabu_list:
                if tabu_idx == candidate_idx:
                    age = self.iteration - tabu_iter
                    tabu_penalty += 0.1 * np.exp(-age / 50)  # Exponential decay
            
            row = self.df.iloc[candidate_idx]
            score, match_score, diversity_score = calculate_final_score(
                candidate_idx, row, self.selected_indices, self.feature_data,
                self.column_mapping, self.ranges, self.params['diversity_weight']
            )
            
            # Apply tabu penalty
            score = max(0, score - tabu_penalty)
            
            if score > best_score:
                best_candidate = candidate_idx
                best_score = score
                best_details = (match_score, diversity_score)
        
        return best_candidate, best_score, best_details
    
    def mcmc_step(self):
        """Single MCMC step with simulated annealing"""
        candidate, candidate_score, details = self.greedy_local_improvement()
        
        if candidate is None:
            return False  # No improvement found
        
        # Acceptance decision
        score_diff = candidate_score - self.current_score
        if score_diff > 0 or np.random.random() < np.exp(score_diff / self.temperature):
            # Accept
            self.selected_indices.append(candidate)
            self.current_score = candidate_score
            
            # Update tabu list
            self.tabu_list.append((candidate, self.iteration))
            if len(self.tabu_list) > self.params['tabu_size']:
                self.tabu_list.pop(0)
            
            # Check if this is a prioritized hit (high match score)
            if details and details[0] > 0.8:  # High match score threshold
                self.hit_iterations.append(self.iteration)
            
            # Update best score
            if candidate_score > self.best_score:
                self.best_score = candidate_score
                self.last_improvement = self.iteration
            
            return True
        
        return False
    
    def update_temperature(self):
        """Update temperature with cooling and reheating"""
        # Check for reheating
        if (self.iteration - self.last_improvement) >= self.params['reheating_window']:
            if np.random.random() < self.params['reheating_alpha']:
                self.temperature = self.params['initial_T'] * self.params['reheating_temp_ratio']
                self.last_improvement = self.iteration
                return
        
        # Regular cooling
        self.temperature *= self.params['cooling_rate']
    
    def run_chain(self):
        """Run the complete MCMC chain"""
        print(f"Starting chain {self.chain_id}")
        
        for i in range(self.params['max_iters']):
            self.iteration = i
            
            accepted = self.mcmc_step()
            
            # Record history
            self.score_history.append(self.current_score)
            self.best_score_history.append(self.best_score)
            self.acceptance_history.append(1 if accepted else 0)
            self.temperature_history.append(self.temperature)
            
            # Update temperature
            self.update_temperature()
            
            # Progress reporting
            if i % 500 == 0:
                print(f"Chain {self.chain_id}: Iteration {i}, Best Score: {self.best_score:.4f}, Selected: {len(self.selected_indices)}")
        
        print(f"Chain {self.chain_id} completed: {len(self.selected_indices)} points selected, best score: {self.best_score:.4f}")

def run_viveka(df, column_mapping, ranges, hyperparams):
    """Run complete yukthi_opusalgorithm with multiple chains"""
    print("Preparing feature data...")
    
    # Prepare feature data for diversity calculation
    feature_cols = [column_mapping[col] for col in ['temperature', 'humidity', 'pressure']]
    feature_data = df[feature_cols].values
    
    # Standardize features for distance calculation
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(feature_data)
    
    print("Starting yukthi_opuschains...")
    chains = []
    
    # Run multiple chains
    for chain_id in range(hyperparams['chains']):
        chain = VivetaChain(df, feature_data, column_mapping, ranges, hyperparams, chain_id)
        chain.run_chain()
        chains.append(chain)
    
    # Combine results from all chains
    all_selected = []
    all_scores = []
    
    for chain in chains:
        for idx in chain.selected_indices:
            if idx not in [x[0] for x in all_selected]:  # Avoid duplicates
                row = df.iloc[idx]
                final_score, match_score, diversity_score = calculate_final_score(
                    idx, row, [], feature_data, column_mapping, ranges, hyperparams['diversity_weight']
                )
                all_selected.append((idx, final_score, match_score, diversity_score))
    
    # Sort by score (best first)
    all_selected.sort(key=lambda x: x[1], reverse=True)
    
    print(f"yukthi_opuscompleted: {len(all_selected)} unique prioritized points found")
    return all_selected, chains

# ============================================================================
# GROUND TRUTH FILTERING
# ============================================================================

def find_actual_hits(df, column_mapping, ranges):
    """Find all rows that exactly match the specified ranges"""
    conditions = []
    
    for feature, (range_min, range_max) in ranges.items():
        actual_col = column_mapping[feature]
        condition = (df[actual_col] >= range_min) & (df[actual_col] <= range_max)
        conditions.append(condition)
    
    # All conditions must be true
    final_condition = conditions[0]
    for condition in conditions[1:]:
        final_condition = final_condition & condition
    
    actual_hits = df[final_condition].copy()
    actual_hits['hit_type'] = 'actual'
    
    print(f"Found {len(actual_hits)} actual hits (exact matches)")
    return actual_hits

# ============================================================================
# EXPORT AND VISUALIZATION
# ============================================================================

def export_results(viveka_results, actual_hits, df, column_mapping):
    """Export results to CSV files"""
    
    # Export yukthi_opusprioritized results
    viveka_data = []
    for rank, (idx, final_score, match_score, diversity_score) in enumerate(viveka_results, 1):
        row = df.iloc[idx]
        result_row = {
            'rank': rank,
            'id': idx,
            'temperature': row[column_mapping['temperature']],
            'humidity': row[column_mapping['humidity']],
            'pressure': row[column_mapping['pressure']],
            'final_score': final_score,
            'match_score': match_score,
            'diversity_score': diversity_score
        }
        
        # Add other columns
        for col in df.columns:
            if col not in column_mapping.values():
                result_row[col] = row[col]
        
        viveka_data.append(result_row)
    
    viveka_df = pd.DataFrame(viveka_data)
    viveka_df.to_csv('viveka_prioritized.csv', index=False)
    print(f"Exported viveka_prioritized.csv ({len(viveka_df)} rows)")
    
    # Export actual hits
    actual_hits.to_csv('actual_hits.csv', index=False)
    print(f"Exported actual_hits.csv ({len(actual_hits)} rows)")
    
    return viveka_df

def create_visualizations(viveka_results, actual_hits, df, chains, column_mapping):
    """Create all required visualizations"""
    
    # 1. Chain traces plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('yukthi_opusChain Traces', fontsize=16)
    
    for i, chain in enumerate(chains):
        ax = axes[i//2, i%2]
        iterations = range(len(chain.best_score_history))
        ax.plot(iterations, chain.best_score_history, 'b-', alpha=0.7, label='Best Score')
        ax.plot(iterations, chain.score_history, 'g-', alpha=0.3, label='Current Score')
        
        # Mark hit discoveries
        for hit_iter in chain.hit_iterations:
            ax.axvline(x=hit_iter, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title(f'Chain {i}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chains.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Scatter plot of temperature/humidity/pressure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    temp_col = column_mapping['temperature']
    humid_col = column_mapping['humidity']
    press_col = column_mapping['pressure']
    
    # Sample data if too large
    if len(df) > 5000:
        sample_df = df.sample(5000)
        print("Downsampled data for visualization")
    else:
        sample_df = df
    
    # Plot pairs
    pairs = [(temp_col, humid_col, 'Temperature vs Humidity'),
             (temp_col, press_col, 'Temperature vs Pressure'),
             (humid_col, press_col, 'Humidity vs Pressure')]
    
    for i, (x_col, y_col, title) in enumerate(pairs):
        ax = axes[i]
        
        # Plot all points
        ax.scatter(sample_df[x_col], sample_df[y_col], alpha=0.3, s=10, c='lightblue', label='All data')
        
        # Plot actual hits
        if len(actual_hits) > 0:
            ax.scatter(actual_hits[x_col], actual_hits[y_col], alpha=0.8, s=30, c='red', 
                      marker='s', label=f'Actual hits ({len(actual_hits)})')
        
        # Plot yukthi_opusprioritized
        viveka_indices = [x[0] for x in viveka_results[:100]]  # Top 100
        viveka_points = df.iloc[viveka_indices]
        ax.scatter(viveka_points[x_col], viveka_points[y_col], alpha=0.8, s=20, c='green', 
                  marker='^', label=f'yukthi_opustop 100')
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scatter_hits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Acceptance rate and temperature schedule
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Acceptance rates
    for i, chain in enumerate(chains):
        # Calculate rolling acceptance rate
        window = 100
        acceptance_rate = pd.Series(chain.acceptance_history).rolling(window).mean()
        axes[0].plot(acceptance_rate, label=f'Chain {i}', alpha=0.7)
    
    axes[0].set_title('Acceptance Rate vs Iteration')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Acceptance Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temperature schedules
    for i, chain in enumerate(chains):
        axes[1].plot(chain.temperature_history, label=f'Chain {i}', alpha=0.7)
    
    axes[1].set_title('Temperature Schedule vs Iteration')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Temperature')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('acceptance_temp.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Time series (if timestamp available)
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if timestamp_cols:
        print(f"Found timestamp columns: {timestamp_cols}")
        try:
            timestamp_col = timestamp_cols[0]
            df_time = df.copy()
            df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
            df_time = df_time.sort_values(timestamp_col)
            
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))
            
            features = ['temperature', 'humidity', 'pressure']
            for i, feature in enumerate(features):
                actual_col = column_mapping[feature]
                ax = axes[i]
                
                # Plot all data
                ax.plot(df_time[timestamp_col], df_time[actual_col], alpha=0.3, c='lightblue', label='All data')
                
                # Plot actual hits
                if len(actual_hits) > 0:
                    hits_time = actual_hits.copy()
                    hits_time[timestamp_col] = pd.to_datetime(hits_time[timestamp_col])
                    ax.scatter(hits_time[timestamp_col], hits_time[actual_col], 
                              c='red', s=30, alpha=0.8, label='Actual hits')
                
                # Plot yukthi_opusprioritized
                viveka_indices = [x[0] for x in viveka_results[:50]]
                viveka_time = df_time.iloc[viveka_indices]
                ax.scatter(viveka_time[timestamp_col], viveka_time[actual_col], 
                          c='green', s=20, alpha=0.8, label='yukthi_opusprioritized', marker='^')
                
                ax.set_ylabel(feature.title())
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                if i == len(features) - 1:
                    ax.set_xlabel('Time')
            
            plt.suptitle('Time Series: Prioritized and Actual Hits Over Time')
            plt.tight_layout()
            plt.savefig('time_hits.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Could not create time series plot: {e}")
    else:
        print("No timestamp columns found - skipping time series plot")

# ============================================================================
# ANALYSIS AND REPORTING
# ============================================================================

def analyze_results(viveka_results, actual_hits, df):
    """Analyze and report on the results"""
    
    viveka_indices = set([x[0] for x in viveka_results])
    actual_indices = set(actual_hits.index.tolist())
    
    # Calculate overlap
    overlap = viveka_indices.intersection(actual_indices)
    
    # Precision and Recall
    precision = len(overlap) / len(viveka_indices) if len(viveka_indices) > 0 else 0
    recall = len(overlap) / len(actual_indices) if len(actual_indices) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    report = f"""
# yukthi_opusAnalysis Report

## Dataset Overview
- Total datapoints: {len(df):,}
- yukthi_opusprioritized points: {len(viveka_results):,}
- Actual hits (ground truth): {len(actual_hits):,}
- Overlap (prioritized ∩ actual): {len(overlap):,}

## Performance Metrics
- **Precision**: {precision:.3f} ({len(overlap)}/{len(viveka_indices)})
- **Recall**: {recall:.3f} ({len(overlap)}/{len(actual_indices)})
- **F1 Score**: {f1_score:.3f}

## Hyperparameters Used
- Chains: {HYPERPARAMS['chains']}
- Max iterations: {HYPERPARAMS['max_iters']:,}
- Initial temperature: {HYPERPARAMS['initial_T']}
- Cooling rate: {HYPERPARAMS['cooling_rate']}
- Diversity weight (λ): {HYPERPARAMS['diversity_weight']}
- Neighborhood size: {HYPERPARAMS['neighborhood_size']}

## Scoring Formula
- Feature score: 1 if in range, else max(0, 1 - distance/span)
- Match score: mean(temp_score, humid_score, pressure_score)
- Diversity score: normalized min_distance_to_selected_points
- Final score: match_score × (1-λ) + diversity_score × λ

## Key Findings
- yukthi_opussuccessfully identified {len(overlap)} out of {len(actual_indices)} actual hits
- The algorithm balanced exploration vs exploitation using diversity weighting
- {"High precision indicates good specificity" if precision > 0.7 else "Low precision suggests over-exploration"}
- {"High recall indicates good coverage" if recall > 0.7 else "Low recall suggests under-exploration"}

## Caveats
- Results depend on hyperparameter tuning and random initialization
- Diversity weighting may reduce precision in favor of exploration
- Performance varies with data distribution and range specifications
"""
    
    print(report)
    
    # Save report
    with open('viveka_report.md', 'w') as f:
        f.write(report)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'overlap_count': len(overlap),
        'viveka_count': len(viveka_results),
        'actual_count': len(actual_hits)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("VIVEKA: MCMC + Greedy + Simulated Annealing for Datapoint Prioritization")
    print("=" * 80)
    
    # Load and validate data
    df, column_mapping = load_and_validate_data(CSV_FILE)
    if df is None:
        return
    
    print(f"\nDetected columns:")
    for feature, actual_col in column_mapping.items():
        print(f"  {feature}: {actual_col}")
    
    print(f"\nTarget ranges:")
    for feature, (min_val, max_val) in RANGES.items():
        print(f"  {feature}: [{min_val}, {max_val}]")
    
    print(f"\nData shape: {df.shape}")
    print("Data preview:")
    print(df[[column_mapping[f] for f in ['temperature', 'humidity', 'pressure']]].head())
    
    # Ask for confirmation
    response = input("\nProceed with yukthi_opusanalysis? (type 'GO' to continue): ").strip().upper()
    if response != 'GO':
        print("Analysis cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("STARTING yukthi_opusANALYSIS")
    print("=" * 80)
    
    # Run yukthi_opusalgorithm
    viveka_results, chains = run_viveka(df, column_mapping, RANGES, HYPERPARAMS)
    
    # Find actual hits (ground truth)
    actual_hits = find_actual_hits(df, column_mapping, RANGES)
    
    # Export results
    viveka_df = export_results(viveka_results, actual_hits, df, column_mapping)
    
    # Create visualizations
    create_visualizations(viveka_results, actual_hits, df, chains, column_mapping)
    
    # Analyze and report
    metrics = analyze_results(viveka_results, actual_hits, df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Files generated:")
    print("  - viveka_prioritized.csv (ranked results)")
    print("  - actual_hits.csv (ground truth matches)")
    print("  - chains.png (MCMC chain traces)")
    print("  - scatter_hits.png (feature space visualization)")
    print("  - acceptance_temp.png (algorithm diagnostics)")
    print("  - time_hits.png (time series, if applicable)")
    print("  - viveka_report.md (analysis summary)")
    
    return viveka_results, actual_hits, chains, metrics

# ============================================================================
# READY TO RUN
# ============================================================================

print("yukthi_opusimplementation ready!")
print("1. Upload your DATA.csv file")
print("2. Modify the RANGES dictionary above with your target values")
print("3. Optionally adjust HYPERPARAMS")
print("4. Run main() to start the analysis")
print("\nExample:")
print("RANGES = {'temperature': [20, 28], 'humidity': [40, 70], 'pressure': [1008, 1015]}")

# Uncomment the next line to run automatically (after setting ranges and uploading data)
main()