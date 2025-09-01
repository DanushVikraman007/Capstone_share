#!/usr/bin/env python3
"""
Fast YO (Yukti Opis) Hybrid Optimization Framework
Optimized for Speed: Vectorized operations, reduced iterations, smart sampling

Author: Fast YO Framework Implementation
Version: 2.0 (Speed Optimized)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class FastYO:
    """
    Fast YO Framework - Speed Optimized Version
    Uses vectorized operations and smart sampling for 10x+ speedup
    """
    
    def __init__(self, csv_file, target_ranges):
        self.csv_file = csv_file
        self.target_ranges = target_ranges
        self.df = None
        self.feature_data = None
        self.column_mapping = {}
        self.ground_truth_indices = []
        self.yo_candidates = []
        self.execution_time = 0
        
        # Fast hyperparameters (reduced for speed)
        self.fast_params = None
        
        # Load data
        self._load_data()
        self._set_fast_params()
        
        print(f"FastYO initialized: {len(self.df)} rows, {len(self.df.columns)} cols")
        
    def _load_data(self):
        """Fast data loading with minimal validation"""
        self.df = pd.read_csv(self.csv_file)
        
        # Quick column mapping
        required = ['temperature', 'humidity', 'pressure']
        for feature in required:
            for col in self.df.columns:
                if feature.lower() in col.lower():
                    self.column_mapping[feature] = col
                    break
        
        # Drop NaN rows quickly
        feature_cols = list(self.column_mapping.values())
        self.df = self.df.dropna(subset=feature_cols)
        
        # Prepare standardized features (vectorized)
        scaler = StandardScaler()
        self.feature_data = scaler.fit_transform(self.df[feature_cols].values)
        
    def _set_fast_params(self):
        """Optimized hyperparameters for speed"""
        N = len(self.df)
        self.fast_params = {
   
    'chains': max(20, N // 1200),         # Very wide exploration
    'max_iters': max(6000, N // 100),     # Moderate length
    'neighborhood_size': max(256, N // 60), # Large jump capacity
    'sample_size': N,                     # All data
    'diversity_weight': 0.0,             # Almost no penalty for similarity
    'initial_T': 3,                     # Very high acceptance early
    'cooling_rate': 0.99,                 # Very slow cooling → exploration dominates
    'tabu_size': 10                        # Small tabu → can revisit promising areas


        }
        print(f"Fast params: {self.fast_params}")
    
    def find_ground_truth_fast(self):
        """Vectorized ground truth finding"""
        print("Finding ground truth (vectorized)...")
        
        # Vectorized range checking
        conditions = []
        for feature, (min_val, max_val) in self.target_ranges.items():
            col = self.column_mapping[feature]
            condition = (self.df[col] >= min_val) & (self.df[col] <= max_val)
            conditions.append(condition)
        
        # Combine all conditions
        final_condition = np.logical_and.reduce(conditions)
        self.ground_truth_indices = self.df[final_condition].index.tolist()
        
        print(f"Ground truth: {len(self.ground_truth_indices)} hits")
        return self.ground_truth_indices
    
    def _vectorized_scoring(self, indices):
        """Vectorized scoring for multiple candidates"""
        if len(indices) == 0:
            return np.array([]), np.array([])
            
        # Get feature values for all candidates at once
        feature_cols = [self.column_mapping[f] for f in ['temperature', 'humidity', 'pressure']]
        candidate_features = self.df.iloc[indices][feature_cols].values
        
        # Vectorized range scoring
        match_scores = []
        for i, feature in enumerate(['temperature', 'humidity', 'pressure']):
            min_val, max_val = self.target_ranges[feature]
            values = candidate_features[:, i]
            
            # Vectorized scoring
            in_range = (values >= min_val) & (values <= max_val)
            range_span = max_val - min_val
            
            # Distance penalty for out-of-range
            below_range = values < min_val
            above_range = values > max_val
            
            scores = np.ones(len(values))  # Start with perfect scores
            scores[below_range] = np.maximum(0, 1 - (min_val - values[below_range]) / range_span)
            scores[above_range] = np.maximum(0, 1 - (values[above_range] - max_val) / range_span)
            
            match_scores.append(scores)
        
        # Average across features
        final_match_scores = np.mean(match_scores, axis=0)
        
        # Simple diversity scores (distance to dataset center)
        center = np.mean(self.feature_data, axis=0)
        candidate_std_features = self.feature_data[indices]
        distances = np.linalg.norm(candidate_std_features - center, axis=1)
        diversity_scores = np.minimum(1.0, distances / np.sqrt(3))  # Normalize
        
        return final_match_scores, diversity_scores
    
    def run_fast_optimization(self):
        """Fast optimization using smart sampling and vectorization"""
        print("Starting FastYO optimization...")
        start_time = time.time()
        
        # Work on sample if dataset is large
        if len(self.df) > self.fast_params['sample_size']:
            print(f"Sampling {self.fast_params['sample_size']} points for optimization...")
            sample_indices = np.random.choice(len(self.df), self.fast_params['sample_size'], 
                                            replace=False)
            working_df = self.df.iloc[sample_indices]
            index_mapping = {i: sample_indices[i] for i in range(len(sample_indices))}
        else:
            working_df = self.df
            sample_indices = np.arange(len(self.df))
            index_mapping = {i: i for i in range(len(self.df))}
        
        print(f"Working with {len(working_df)} datapoints")
        
        # Fast multi-chain optimization
        all_candidates = set()
        chain_results = []
        
        for chain_id in range(self.fast_params['chains']):
            print(f"Chain {chain_id}...")
            
            # Initialize chain
            current_candidates = set()
            temperature = self.fast_params['initial_T']
            best_score = 0
            
            for iteration in range(self.fast_params['max_iters']):
                # Get random neighborhood
                neighborhood_size = min(self.fast_params['neighborhood_size'], len(working_df))
                candidates = np.random.choice(len(working_df), neighborhood_size, replace=False)
                
                # Remove already selected
                candidates = [c for c in candidates if c not in current_candidates]
                if not candidates:
                    continue
                
                # Vectorized scoring
                match_scores, diversity_scores = self._vectorized_scoring(
                    [sample_indices[c] for c in candidates]
                )
                
                if len(match_scores) == 0:
                    continue
                
                # Combined scores
                λ = self.fast_params['diversity_weight']
                final_scores = match_scores * (1 - λ) + diversity_scores * λ
                
                # Find best candidate
                best_idx = np.argmax(final_scores)
                best_candidate = candidates[best_idx]
                candidate_score = final_scores[best_idx]
                
                # Simple acceptance (higher scores more likely)
                if candidate_score > best_score * 0.9:  # Accept if within 90% of best
                    current_candidates.add(best_candidate)
                    all_candidates.add(sample_indices[best_candidate])
                    if candidate_score > best_score:
                        best_score = candidate_score
                
                # Fast cooling
                temperature *= self.fast_params['cooling_rate']
                
                # Early stopping if enough candidates found
                if len(current_candidates) > 20:
                    break
            
            chain_results.append({
                'chain_id': chain_id,
                'candidates': len(current_candidates),
                'best_score': best_score
            })
            
            print(f"  Chain {chain_id}: {len(current_candidates)} candidates, score: {best_score:.3f}")
        
        # Final scoring of all candidates
        if all_candidates:
            all_candidate_list = list(all_candidates)
            final_match_scores, final_diversity_scores = self._vectorized_scoring(all_candidate_list)
            
            # Create results
            λ = self.fast_params['diversity_weight']
            combined_scores = final_match_scores * (1 - λ) + final_diversity_scores * λ
            
            # Sort by score
            sorted_indices = np.argsort(combined_scores)[::-1]  # Descending
            
            self.yo_candidates = []
            for i in sorted_indices:
                idx = all_candidate_list[i]
                self.yo_candidates.append({
                    'index': idx,
                    'final_score': combined_scores[i],
                    'match_score': final_match_scores[i],
                    'diversity_score': final_diversity_scores[i]
                })
        
        self.execution_time = time.time() - start_time
        print(f"FastYO completed in {self.execution_time:.2f}s")
        print(f"Found {len(self.yo_candidates)} candidates")
        
        return chain_results
    
    def analyze_performance(self):
        """Fast performance analysis"""
        if not self.yo_candidates or not self.ground_truth_indices:
            return {}
        
        yo_indices = {c['index'] for c in self.yo_candidates}
        gt_indices = set(self.ground_truth_indices)
        overlap = yo_indices.intersection(gt_indices)
        
        precision = len(overlap) / len(yo_indices) if yo_indices else 0
        recall = len(overlap) / len(gt_indices) if gt_indices else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'yo_candidates': len(yo_indices),
            'ground_truth': len(gt_indices),
            'overlap': len(overlap),
            'execution_time': self.execution_time,
            'throughput': len(self.df) / self.execution_time if self.execution_time > 0 else 0
        }
        
        return metrics


def create_fast_visualizations(yo_framework):
    """Create essential visualizations quickly"""
    print("Creating fast visualizations...")
    
    # Get data
    yo_indices = [c['index'] for c in yo_framework.yo_candidates[:100]]  # Top 100
    gt_indices = yo_framework.ground_truth_indices
    
    temp_col = yo_framework.column_mapping['temperature']
    humid_col = yo_framework.column_mapping['humidity']
    press_col = yo_framework.column_mapping['pressure']
    
    # Single comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FastYO Results Analysis', fontsize=16, fontweight='bold')
    
    # Sample data for speed
    if len(yo_framework.df) > 2000:
        sample_df = yo_framework.df.sample(2000)
    else:
        sample_df = yo_framework.df
    
    # 1. Temperature vs Humidity
    ax = axes[0, 0]
    ax.scatter(sample_df[temp_col], sample_df[humid_col], 
              alpha=0.3, s=5, c='lightgray', label='All data')
    
    if gt_indices:
        gt_df = yo_framework.df.iloc[gt_indices]
        ax.scatter(gt_df[temp_col], gt_df[humid_col], 
                  s=5, c='red', marker='s', alpha=0.8, 
                  label=f'Ground Truth ({len(gt_indices)})')
    
    if yo_indices:
        yo_df = yo_framework.df.iloc[yo_indices]
        ax.scatter(yo_df[temp_col], yo_df[humid_col], 
                  s=6, c='blue', marker='^', alpha=0.9, 
                  edgecolor='black', linewidth=0.7,
                  label=f'YO Candidates ({len(yo_indices)})')
    
    # Target range rectangle
    temp_range = yo_framework.target_ranges['temperature']
    humid_range = yo_framework.target_ranges['humidity']
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((temp_range[0], humid_range[0]), 
                    temp_range[1] - temp_range[0], 
                    humid_range[1] - humid_range[0],
                    linewidth=2, edgecolor='green', 
                    facecolor='green', alpha=0.1, label='Target Range')
    ax.add_patch(rect)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_title('Temperature vs Humidity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Temperature vs Pressure
    ax = axes[0, 1]
    ax.scatter(sample_df[temp_col], sample_df[press_col], 
              alpha=0.3, s=5, c='lightgray', label='All data')
    
    if gt_indices:
        ax.scatter(gt_df[temp_col], gt_df[press_col], 
                  s=5, c='red', marker='s', alpha=0.8, label='Ground Truth')
    
    if yo_indices:
        ax.scatter(yo_df[temp_col], yo_df[press_col], 
                  s=6, c='blue', marker='^', alpha=0.9, label='YO Candidates')
    
    # Target range rectangle
    press_range = yo_framework.target_ranges['pressure']
    rect = Rectangle((temp_range[0], press_range[0]), 
                    temp_range[1] - temp_range[0], 
                    press_range[1] - press_range[0],
                    linewidth=2, edgecolor='green', 
                    facecolor='green', alpha=0.1)
    ax.add_patch(rect)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Pressure')
    ax.set_title('Temperature vs Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Humidity vs Pressure
    ax = axes[1, 0]
    ax.scatter(sample_df[humid_col], sample_df[press_col], 
              alpha=0.3, s=5, c='lightgray', label='All data')
    
    if gt_indices:
        ax.scatter(gt_df[humid_col], gt_df[press_col], 
                  s=5, c='red', marker='s', alpha=0.8, label='Ground Truth')
    
    if yo_indices:
        ax.scatter(yo_df[humid_col], yo_df[press_col], 
                  s=6, c='blue', marker='^', alpha=0.9, label='YO Candidates')
    
    rect = Rectangle((humid_range[0], press_range[0]), 
                    humid_range[1] - humid_range[0], 
                    press_range[1] - press_range[0],
                    linewidth=2, edgecolor='green', 
                    facecolor='green', alpha=0.1)
    ax.add_patch(rect)
    
    ax.set_xlabel('Humidity')
    ax.set_ylabel('Pressure')
    ax.set_title('Humidity vs Pressure')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Score distribution
    ax = axes[1, 1]
    if yo_framework.yo_candidates:
        scores = [c['final_score'] for c in yo_framework.yo_candidates]
        match_scores = [c['match_score'] for c in yo_framework.yo_candidates]
        
        ax.hist(scores, bins=31, alpha=0.7, color='blue', density=True, 
               label=f'Final Scores (μ={np.mean(scores):.3f})')
        ax.hist(match_scores, bins=30, alpha=0.7, color='orange', density=True, 
               label=f'Match Scores (μ={np.mean(match_scores):.3f})')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fastyo_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved: fastyo_results.png")


def export_fast_results(yo_framework):
    """Quick export of results"""
    print("Exporting results...")
    
    # Export YO results
    if yo_framework.yo_candidates:
        results_data = []
        for rank, candidate in enumerate(yo_framework.yo_candidates, 1):
            idx = candidate['index']
            row = yo_framework.df.iloc[idx]
            
            result_row = {
                'rank': rank,
                'original_index': idx,
                'final_score': candidate['final_score'],
                'match_score': candidate['match_score'],
                'diversity_score': candidate['diversity_score'],
                'is_ground_truth': idx in yo_framework.ground_truth_indices,
                'temperature': row[yo_framework.column_mapping['temperature']],
                'humidity': row[yo_framework.column_mapping['humidity']],
                'pressure': row[yo_framework.column_mapping['pressure']]
            }
            results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('fastyo_results.csv', index=False)
        print(f"Exported fastyo_results.csv ({len(results_df)} rows)")
    
    # Export ground truth
    if yo_framework.ground_truth_indices:
        gt_df = yo_framework.df.iloc[yo_framework.ground_truth_indices].copy()
        gt_df.to_csv('fastyo_ground_truth.csv', index=False)
        print(f"Exported fastyo_ground_truth.csv ({len(gt_df)} rows)")


def print_fast_summary(yo_framework, metrics):
    """Print concise performance summary"""
    print("\n" + "="*60)
    print("FASTYO PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Dataset size: {len(yo_framework.df):,} rows")
    print(f"Execution time: {metrics['execution_time']:.2f} seconds")
    print(f"Throughput: {metrics['throughput']:.0f} records/second")
    print()
    print("RESULTS:")
    print(f"  YO Candidates: {metrics['yo_candidates']:,}")
    print(f"  Ground Truth: {metrics['ground_truth']:,}")
    print(f"  Overlap: {metrics['overlap']:,}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print()
    print("TOP 10 FASTYO CANDIDATES:")
    
    for i, candidate in enumerate(yo_framework.yo_candidates[:10]):
        idx = candidate['index']
        row = yo_framework.df.iloc[idx]
        is_gt = "✓" if idx in yo_framework.ground_truth_indices else "✗"
        
        temp = row[yo_framework.column_mapping['temperature']]
        humid = row[yo_framework.column_mapping['humidity']]
        press = row[yo_framework.column_mapping['pressure']]
        
        print(f"  {i+1:2d}. Row {idx:5d} | Score: {candidate['final_score']:.3f} | "
              f"T: {temp:6.2f} | H: {humid:.3f} | P: {press:7.2f} | GT: {is_gt}")
    
    print("="*60)
    print("Files: fastyo_results.csv, fastyo_ground_truth.csv, fastyo_results.png")


def run_fast_yo(csv_file, target_ranges):
    """
    Complete FastYO pipeline - optimized for speed
    
    Args:
        csv_file (str): Path to CSV file
        target_ranges (dict): Target ranges for features
    
    Returns:
        FastYO framework instance
    """
    
    print("="*60)
    print("FASTYO - SPEED OPTIMIZED YO FRAMEWORK")
    print("="*60)
    
    try:
        # Initialize framework
        yo = FastYO(csv_file, target_ranges)
        
        # Find ground truth
        yo.find_ground_truth_fast()
        
        # Run optimization
        chain_results = yo.run_fast_optimization()
        
        # Analyze performance
        metrics = yo.analyze_performance()
        
        # Create visualizations
        create_fast_visualizations(yo)
        
        # Export results
        export_fast_results(yo)
        
        # Print summary
        print_fast_summary(yo, metrics)
        
        print(f"\nFastYO completed in {yo.execution_time:.2f} seconds!")
        return yo
        
    except Exception as e:
        print(f"Error in FastYO: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("FastYO - Speed Optimized YO Framework")
    print("="*50)
    
    # Example usage
    target_ranges = {
        'temperature': [15, 20],    # °C  
        'humidity': [0.3, 0.7],     # decimal
        'pressure': [1013, 1017]    # hPa
    }
    
    print("\nExample usage:")
    print("yo = run_fast_yo('DATA.csv', target_ranges)")
    print("\nReady to run! Modify target_ranges above and call run_fast_yo()")
    
    # Uncomment to run:
    yo = run_fast_yo('DATA.csv', target_ranges)