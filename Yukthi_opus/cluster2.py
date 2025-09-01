#!/usr/bin/env python3
"""
Ultra-Fast YO (Yukti Opis) Hybrid Optimization Framework with Smart Clustering
Optimized for Maximum Speed + Intelligent Sampling

Author: Ultra-Fast YO Framework Implementation
Version: 3.0 (Ultra Speed + Smart Sampling)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class UltraFastYO:
    """
    Ultra-Fast YO Framework - Maximum Speed with Smart Clustering
    Uses adaptive sampling, hierarchical clustering, and intelligent data reduction
    """
    
    def __init__(self, csv_file, target_ranges, clustering_params=None, speed_params=None):
        self.csv_file = csv_file
        self.target_ranges = target_ranges
        self.df = None
        self.feature_data = None
        self.sample_df = None
        self.sample_feature_data = None
        self.column_mapping = {}
        self.ground_truth_indices = []
        self.yo_candidates = []
        self.execution_time = 0
        
        # Smart clustering parameters
        self.clustering_params = clustering_params or {
            'enable_clustering': True,
            'sample_ratio': 0.05,           # Only use 5% of data for clustering analysis
            'min_sample_size': 1000,        # Minimum sample size
            'max_sample_size': 5000,        # Maximum sample size for clustering
            'cluster_radius': 0.15,         # Radius for clustering
            'min_cluster_size': 3,          # Minimum points per cluster
            'cluster_bias': 0.5,            # Bias toward cluster centers
            'max_clusters': 8,              # Maximum clusters to track
            'cluster_focus_prob': 0.7,      # Focus probability on clusters
            'use_pca': True,                # Use PCA for dimensionality reduction
            'pca_variance': 0.95            # Variance to retain in PCA
        }
        
        # Ultra-speed parameters
        self.speed_params = speed_params or {
            'max_data_size': 50000,         # Maximum data size to process
            'early_sample_ratio': 0.5,     # Use 15% of data for initial exploration
            'chains': 80,                   # Fewer chains for speed
            'max_iters': 9000,              # Fewer iterations
            'convergence_patience': 100,    # Early stopping patience
            'batch_size': 500,              # Batch processing size
            'vectorization_threshold': 1000 # When to use full vectorization
        }
        
        # Performance tracking
        self.performance_stats = {
            'load_time': 0,
            'cluster_time': 0,
            'optimization_time': 0,
            'total_samples_processed': 0
        }
        
        # Clustering state
        self.cluster_centers = []
        self.cluster_labels = []
        self.pca_transformer = None
        self.sample_indices = []
        
        # Load and prepare data
        self._ultra_fast_data_loading()
        self._smart_clustering_setup()
        
        data_size = len(self.df)
        sample_size = len(self.sample_df) if self.sample_df is not None else data_size
        
        print(f"UltraFastYO initialized: {data_size:,} total rows, {sample_size:,} working rows")
        if self.clustering_params['enable_clustering']:
            print(f"Smart clustering: ENABLED (using {sample_size:,} samples)")
        
    def _ultra_fast_data_loading(self):
        """Ultra-fast data loading with intelligent sampling"""
        start_time = time.time()
        print("Ultra-fast data loading...")
        
        # Load data with chunking for very large files
        try:
            # Try to load all data first
            self.df = pd.read_csv(self.csv_file)
        except MemoryError:
            # If too large, load in chunks and sample
            print("Large file detected, using chunked loading...")
            chunk_size = 10000
            sample_data = []
            total_rows = 0
            
            for chunk in pd.read_csv(self.csv_file, chunksize=chunk_size):
                total_rows += len(chunk)
                # Sample from each chunk
                if len(chunk) > 100:
                    sample_size = max(50, int(len(chunk) * 0.1))
                    sample_data.append(chunk.sample(n=sample_size))
            
            self.df = pd.concat(sample_data, ignore_index=True)
            print(f"Loaded {len(self.df):,} samples from {total_rows:,} total rows")
        
        # Quick column mapping
        required = ['temperature', 'humidity', 'pressure']
        for feature in required:
            for col in self.df.columns:
                if feature.lower() in col.lower():
                    self.column_mapping[feature] = col
                    break
        
        if len(self.column_mapping) != 3:
            raise ValueError(f"Could not find all required columns. Found: {self.column_mapping}")
        
        # Drop NaN rows quickly
        feature_cols = list(self.column_mapping.values())
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=feature_cols)
        print(f"Cleaned data: {len(self.df):,} rows (removed {initial_size - len(self.df):,} NaN rows)")
        
        # Intelligent sampling for processing
        max_size = self.speed_params['max_data_size']
        if len(self.df) > max_size:
            print(f"Large dataset detected. Smart sampling to {max_size:,} rows...")
            # Stratified sampling to maintain distribution
            sample_indices = self._stratified_sample(len(self.df), max_size)
            self.sample_df = self.df.iloc[sample_indices].copy()
            self.sample_indices = sample_indices
        else:
            self.sample_df = self.df.copy()
            self.sample_indices = list(range(len(self.df)))
        
        # Prepare standardized features
        scaler = StandardScaler()
        feature_values = self.sample_df[feature_cols].values
        self.sample_feature_data = scaler.fit_transform(feature_values)
        
        # Also standardize full dataset for final scoring
        if len(self.df) != len(self.sample_df):
            self.feature_data = scaler.transform(self.df[feature_cols].values)
        else:
            self.feature_data = self.sample_feature_data.copy()
        
        self.performance_stats['load_time'] = time.time() - start_time
        print(f"Data loading completed in {self.performance_stats['load_time']:.2f}s")
    
    def _stratified_sample(self, total_size, sample_size):
        """Create stratified sample to maintain data distribution"""
        if sample_size >= total_size:
            return list(range(total_size))
        
        # Simple stratified sampling based on index ranges
        stride = total_size // sample_size
        indices = []
        
        for i in range(sample_size):
            start_idx = i * stride
            end_idx = min(start_idx + stride, total_size)
            # Random selection within each stratum
            if start_idx < total_size:
                idx = np.random.randint(start_idx, min(end_idx, total_size))
                indices.append(idx)
        
        # Add some random samples to fill up to sample_size
        remaining = sample_size - len(indices)
        if remaining > 0:
            all_indices = set(range(total_size))
            used_indices = set(indices)
            available = list(all_indices - used_indices)
            if available:
                additional = np.random.choice(available, 
                                            min(remaining, len(available)), 
                                            replace=False)
                indices.extend(additional)
        
        return sorted(indices[:sample_size])
    
    def _smart_clustering_setup(self):
        """Smart clustering setup using only a small sample of data"""
        if not self.clustering_params['enable_clustering']:
            return
        
        start_time = time.time()
        print("Setting up smart clustering...")
        
        # Determine clustering sample size
        total_samples = len(self.sample_df)
        min_size = self.clustering_params['min_sample_size']
        max_size = self.clustering_params['max_sample_size']
        ratio = self.clustering_params['sample_ratio']
        
        cluster_sample_size = min(max_size, 
                                max(min_size, int(total_samples * ratio)))
        
        if cluster_sample_size >= total_samples:
            clustering_indices = list(range(total_samples))
        else:
            # Smart sampling: include some diverse points
            clustering_indices = self._get_diverse_sample(cluster_sample_size)
        
        print(f"Using {len(clustering_indices):,} samples for clustering analysis")
        
        # Get features for clustering
        clustering_features = self.sample_feature_data[clustering_indices]
        
        # Optional PCA for speed
        if (self.clustering_params['use_pca'] and 
            clustering_features.shape[1] > 2 and 
            len(clustering_indices) > 100):
            
            pca = PCA(n_components=min(3, clustering_features.shape[1]))
            clustering_features = pca.fit_transform(clustering_features)
            self.pca_transformer = pca
            print(f"Applied PCA: {clustering_features.shape[1]} components")
        
        # Fast clustering using MiniBatchKMeans for initial centers
        if len(clustering_indices) > 50:
            # Use KMeans for initial cluster discovery
            n_clusters = min(self.clustering_params['max_clusters'], 
                           max(2, len(clustering_indices) // 20))
            
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, 
                                   batch_size=min(100, len(clustering_indices)),
                                   random_state=42,
                                   n_init=3)  # Fewer initializations for speed
            
            cluster_labels = kmeans.fit_predict(clustering_features)
            
            # Convert KMeans centers back to original space if PCA was used
            if self.pca_transformer is not None:
                cluster_centers = self.pca_transformer.inverse_transform(kmeans.cluster_centers_)
            else:
                cluster_centers = kmeans.cluster_centers_
            
            # Filter out very small clusters
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_clusters = unique_labels[counts >= self.clustering_params['min_cluster_size']]
            
            self.cluster_centers = [cluster_centers[i] for i in valid_clusters]
            
            print(f"Identified {len(self.cluster_centers)} clusters using fast KMeans")
        
        self.performance_stats['cluster_time'] = time.time() - start_time
        print(f"Clustering setup completed in {self.performance_stats['cluster_time']:.2f}s")
    
    def _get_diverse_sample(self, sample_size):
        """Get a diverse sample of indices for clustering analysis"""
        total_size = len(self.sample_feature_data)
        
        if sample_size >= total_size:
            return list(range(total_size))
        
        # Use a simple diversity-based sampling
        # Start with random sample
        initial_sample = np.random.choice(total_size, 
                                        min(sample_size // 2, total_size), 
                                        replace=False)
        
        selected_indices = set(initial_sample)
        
        # Add diverse points
        remaining_size = sample_size - len(selected_indices)
        if remaining_size > 0:
            # Get features of already selected points
            selected_features = self.sample_feature_data[list(selected_indices)]
            selected_center = np.mean(selected_features, axis=0)
            
            # Find points far from current selection
            all_indices = np.array(range(total_size))
            unselected_mask = ~np.isin(all_indices, list(selected_indices))
            unselected_indices = all_indices[unselected_mask]
            
            if len(unselected_indices) > 0:
                unselected_features = self.sample_feature_data[unselected_indices]
                distances = np.linalg.norm(unselected_features - selected_center, axis=1)
                
                # Select points with higher distance probability
                probabilities = distances / (distances.sum() + 1e-8)
                
                additional_count = min(remaining_size, len(unselected_indices))
                additional_indices = np.random.choice(
                    unselected_indices,
                    additional_count,
                    replace=False,
                    p=probabilities
                )
                
                selected_indices.update(additional_indices)
        
        return list(selected_indices)
    
    def find_ground_truth_fast(self):
        """Ultra-fast ground truth finding with smart sampling"""
        print("Finding ground truth (ultra-fast)...")
        start_time = time.time()
        
        # Work on full dataset for ground truth (vectorized)
        conditions = []
        for feature, (min_val, max_val) in self.target_ranges.items():
            col = self.column_mapping[feature]
            condition = (self.df[col] >= min_val) & (self.df[col] <= max_val)
            conditions.append(condition)
        
        # Combine all conditions
        final_condition = np.logical_and.reduce(conditions)
        self.ground_truth_indices = self.df[final_condition].index.tolist()
        
        print(f"Ground truth: {len(self.ground_truth_indices)} hits "
              f"({time.time() - start_time:.2f}s)")
        return self.ground_truth_indices
    
    def _get_smart_neighborhood(self, current_candidates, neighborhood_size):
        """Get smart neighborhood using clustering bias and exploration"""
        working_size = len(self.sample_df)
        
        if neighborhood_size >= working_size:
            return list(range(working_size))
        
        # Decide between cluster-focused and exploration
        use_clusters = (self.clustering_params['enable_clustering'] and 
                       self.cluster_centers and
                       np.random.random() < self.clustering_params['cluster_focus_prob'])
        
        if use_clusters:
            return self._get_cluster_focused_candidates(neighborhood_size, current_candidates)
        else:
            return self._get_exploration_candidates(neighborhood_size, current_candidates)
    
    def _get_cluster_focused_candidates(self, sample_size, exclude_set):
        """Get candidates biased toward cluster regions"""
        candidates = set()
        samples_per_cluster = max(1, sample_size // len(self.cluster_centers))
        
        for center in self.cluster_centers:
            # Find closest points to cluster center
            distances = np.linalg.norm(self.sample_feature_data - center, axis=1)
            
            # Get indices sorted by distance
            sorted_indices = np.argsort(distances)
            
            # Select from closest points, avoiding already selected
            selected_count = 0
            for idx in sorted_indices:
                if idx not in exclude_set and idx not in candidates:
                    candidates.add(idx)
                    selected_count += 1
                    if selected_count >= samples_per_cluster:
                        break
        
        # Fill remaining slots with random selection
        remaining = sample_size - len(candidates)
        if remaining > 0:
            all_indices = set(range(len(self.sample_df)))
            available = list(all_indices - candidates - exclude_set)
            if available:
                additional = np.random.choice(available, 
                                            min(remaining, len(available)), 
                                            replace=False)
                candidates.update(additional)
        
        return list(candidates)
    
    def _get_exploration_candidates(self, sample_size, exclude_set):
        """Get diverse exploration candidates"""
        all_indices = set(range(len(self.sample_df)))
        available = list(all_indices - exclude_set)
        
        if len(available) <= sample_size:
            return available
        
        return list(np.random.choice(available, sample_size, replace=False))
    
    def _ultra_fast_scoring(self, indices):
        """Ultra-fast vectorized scoring with clustering bonuses"""
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert sample indices to original dataset indices if needed
        if len(self.df) != len(self.sample_df):
            original_indices = [self.sample_indices[i] for i in indices]
        else:
            original_indices = indices
        
        # Get feature values (vectorized)
        feature_cols = [self.column_mapping[f] for f in ['temperature', 'humidity', 'pressure']]
        candidate_features = self.df.iloc[original_indices][feature_cols].values
        
        # Ultra-fast range scoring
        match_scores = np.ones(len(indices))
        for i, feature in enumerate(['temperature', 'humidity', 'pressure']):
            min_val, max_val = self.target_ranges[feature]
            values = candidate_features[:, i]
            range_span = max_val - min_val
            
            # Vectorized scoring with clipping for speed
            range_distances = np.maximum(0, np.maximum(min_val - values, values - max_val))
            feature_scores = np.maximum(0, 1 - range_distances / range_span)
            match_scores *= feature_scores  # Multiplicative combination
        
        # Fast diversity scores
        if len(self.sample_df) == len(self.df):
            candidate_std_features = self.feature_data[original_indices]
        else:
            candidate_std_features = self.sample_feature_data[indices]
        
        center = np.mean(self.sample_feature_data, axis=0)
        distances = np.linalg.norm(candidate_std_features - center, axis=1)
        diversity_scores = np.minimum(1.0, distances / 2.0)  # Normalized diversity
        
        # Fast clustering scores
        clustering_scores = np.zeros(len(indices))
        if self.clustering_params['enable_clustering'] and self.cluster_centers:
            for center in self.cluster_centers:
                cluster_distances = np.linalg.norm(candidate_std_features - center, axis=1)
                cluster_bonuses = np.exp(-cluster_distances)
                clustering_scores = np.maximum(clustering_scores, cluster_bonuses)
        
        return match_scores, diversity_scores, clustering_scores
    
    def run_ultra_fast_optimization(self):
        """Ultra-fast optimization with intelligent early stopping"""
        print("Starting ultra-fast optimization...")
        start_time = time.time()
        
        # Adaptive parameters based on data size
        data_size = len(self.sample_df)
        chains = min(self.speed_params['chains'], data_size // 100)
        max_iters = min(self.speed_params['max_iters'], data_size * 2)
        
        print(f"Running {chains} chains with max {max_iters} iterations each")
        
        all_candidates = set()
        best_global_score = 0
        convergence_count = 0
        
        # Multi-chain optimization
        for chain_id in range(chains):
            current_candidates = set()
            temperature = 2.0  # Lower initial temperature for speed
            best_chain_score = 0
            stagnation = 0
            
            for iteration in range(max_iters):
                # Adaptive neighborhood size
                base_size = min(200, data_size // 10)
                neighborhood_size = max(50, base_size - iteration // 100)
                
                # Get smart neighborhood
                candidates = self._get_smart_neighborhood(
                    current_candidates, neighborhood_size
                )
                
                if not candidates:
                    break
                
                # Ultra-fast scoring
                match_scores, diversity_scores, clustering_scores = (
                    self._ultra_fast_scoring(candidates)
                )
                
                if len(match_scores) == 0:
                    continue
                
                # Fast score combination
                λ = 0.1  # Low diversity weight for speed
                cluster_weight = self.clustering_params['cluster_bias']
                
                final_scores = (match_scores * 0.8 + 
                              diversity_scores * λ + 
                              clustering_scores * cluster_weight)
                
                # Select best candidates
                n_select = min(5, len(candidates))  # Select multiple per iteration
                best_indices = np.argsort(final_scores)[-n_select:]
                
                for idx in best_indices:
                    candidate_idx = candidates[idx]
                    score = final_scores[idx]
                    
                    # Fast acceptance criterion
                    if score > best_chain_score * 0.9 or np.random.random() < 0.3:
                        current_candidates.add(candidate_idx)
                        all_candidates.add(self.sample_indices[candidate_idx] if 
                                         len(self.df) != len(self.sample_df) else candidate_idx)
                        
                        if score > best_chain_score:
                            best_chain_score = score
                            stagnation = 0
                
                stagnation += 1
                temperature *= 0.99  # Fast cooling
                
                # Early stopping
                if (stagnation > self.speed_params['convergence_patience'] or 
                    len(current_candidates) > 50):
                    break
            
            # Global convergence check
            if best_chain_score > best_global_score:
                best_global_score = best_chain_score
                convergence_count = 0
            else:
                convergence_count += 1
            
            # Early global stopping
            if convergence_count > chains // 3:
                print(f"Early convergence at chain {chain_id}")
                break
        
        # Final candidate processing
        if all_candidates:
            candidate_list = list(all_candidates)
            
            # Score all final candidates on original dataset
            final_match_scores, final_diversity_scores, final_clustering_scores = (
                self._ultra_fast_scoring_original(candidate_list)
            )
            
            # Combined scoring
            λ = 0.1
            cluster_weight = self.clustering_params['cluster_bias']
            combined_scores = (final_match_scores * 0.8 + 
                             final_diversity_scores * λ + 
                             final_clustering_scores * cluster_weight)
            
            # Sort and create results
            sorted_indices = np.argsort(combined_scores)[::-1]
            
            self.yo_candidates = []
            for i in sorted_indices:
                idx = candidate_list[i]
                self.yo_candidates.append({
                    'index': idx,
                    'final_score': combined_scores[i],
                    'match_score': final_match_scores[i],
                    'diversity_score': final_diversity_scores[i],
                    'clustering_score': final_clustering_scores[i]
                })
        
        self.execution_time = time.time() - start_time
        self.performance_stats['optimization_time'] = self.execution_time
        self.performance_stats['total_samples_processed'] = len(self.sample_df)
        
        print(f"Ultra-fast optimization completed in {self.execution_time:.2f}s")
        print(f"Found {len(self.yo_candidates)} candidates")
        
        return self.yo_candidates
    
    def _ultra_fast_scoring_original(self, indices):
        """Score candidates on original dataset"""
        feature_cols = [self.column_mapping[f] for f in ['temperature', 'humidity', 'pressure']]
        candidate_features = self.df.iloc[indices][feature_cols].values
        
        # Range scoring
        match_scores = np.ones(len(indices))
        for i, feature in enumerate(['temperature', 'humidity', 'pressure']):
            min_val, max_val = self.target_ranges[feature]
            values = candidate_features[:, i]
            range_span = max_val - min_val
            
            range_distances = np.maximum(0, np.maximum(min_val - values, values - max_val))
            feature_scores = np.maximum(0, 1 - range_distances / range_span)
            match_scores *= feature_scores
        
        # Diversity and clustering scores
        candidate_std_features = self.feature_data[indices]
        center = np.mean(self.feature_data, axis=0)
        distances = np.linalg.norm(candidate_std_features - center, axis=1)
        diversity_scores = np.minimum(1.0, distances / 2.0)
        
        clustering_scores = np.zeros(len(indices))
        if self.clustering_params['enable_clustering'] and self.cluster_centers:
            for center in self.cluster_centers:
                cluster_distances = np.linalg.norm(candidate_std_features - center, axis=1)
                cluster_bonuses = np.exp(-cluster_distances)
                clustering_scores = np.maximum(clustering_scores, cluster_bonuses)
        
        return match_scores, diversity_scores, clustering_scores
    
    def analyze_performance(self):
        """Analyze performance with speed metrics"""
        if not self.yo_candidates or not self.ground_truth_indices:
            return {}
        
        yo_indices = {c['index'] for c in self.yo_candidates}
        gt_indices = set(self.ground_truth_indices)
        overlap = yo_indices.intersection(gt_indices)
        
        precision = len(overlap) / len(yo_indices) if yo_indices else 0
        recall = len(overlap) / len(gt_indices) if gt_indices else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_time = (self.performance_stats['load_time'] + 
                     self.performance_stats['cluster_time'] + 
                     self.performance_stats['optimization_time'])
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'yo_candidates': len(yo_indices),
            'ground_truth': len(gt_indices),
            'overlap': len(overlap),
            'execution_time': total_time,
            'load_time': self.performance_stats['load_time'],
            'cluster_time': self.performance_stats['cluster_time'],
            'optimization_time': self.performance_stats['optimization_time'],
            'throughput': len(self.df) / total_time if total_time > 0 else 0,
            'samples_processed': self.performance_stats['total_samples_processed'],
            'sampling_ratio': len(self.sample_df) / len(self.df),
            'clustering_enabled': self.clustering_params['enable_clustering'],
            'clusters_found': len(self.cluster_centers)
        }
        
        return metrics


def create_ultra_fast_visualizations(yo_framework):
    """Create fast visualizations optimized for speed"""
    print("Creating ultra-fast visualizations...")
    
    # Sample data for visualization speed
    max_viz_points = 2000
    total_points = len(yo_framework.df)
    
    if total_points > max_viz_points:
        viz_sample_indices = np.random.choice(total_points, max_viz_points, replace=False)
        viz_df = yo_framework.df.iloc[viz_sample_indices]
    else:
        viz_df = yo_framework.df
        viz_sample_indices = range(total_points)
    
    # Get important points
    yo_indices = [c['index'] for c in yo_framework.yo_candidates[:100]]
    gt_indices = yo_framework.ground_truth_indices
    
    temp_col = yo_framework.column_mapping['temperature']
    humid_col = yo_framework.column_mapping['humidity']
    press_col = yo_framework.column_mapping['pressure']
    
    # Fast plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ultra-Fast YO Results - Speed Optimized', fontsize=16, fontweight='bold')
    
    # 1. Temperature vs Humidity
    ax = axes[0, 0]
    ax.scatter(viz_df[temp_col], viz_df[humid_col], alpha=0.3, s=2, c='lightgray')
    
    if gt_indices:
        gt_df = yo_framework.df.iloc[gt_indices]
        ax.scatter(gt_df[temp_col], gt_df[humid_col], s=10, c='red', marker='.', 
                  alpha=0.6, label=f'Ground Truth ({len(gt_indices)})')
    
    if yo_indices:
        yo_df = yo_framework.df.iloc[yo_indices]
        ax.scatter(yo_df[temp_col], yo_df[humid_col], s=20, c='blue', marker='.', 
                  alpha=0.8, label=f'YO Candidates ({len(yo_indices)})')
    
    # Target range
    temp_range = yo_framework.target_ranges['temperature']
    humid_range = yo_framework.target_ranges['humidity']
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((temp_range[0], humid_range[0]), 
                    temp_range[1] - temp_range[0], 
                    humid_range[1] - humid_range[0],
                    linewidth=2, edgecolor='green', facecolor='green', alpha=0.1)
    ax.add_patch(rect)
    
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Humidity')
    ax.set_title('Temperature vs Humidity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Score distribution
    ax = axes[0, 1]
    if yo_framework.yo_candidates:
        scores = [c['final_score'] for c in yo_framework.yo_candidates]
        ax.hist(scores, bins=30, alpha=0.7, color='blue', density=True)
        ax.set_xlabel('Final Score')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution')
        ax.grid(True, alpha=0.3)
    
    # 3. Performance metrics
    ax = axes[1, 0]
    metrics = yo_framework.analyze_performance()
    
    metric_names = ['Precision', 'Recall', 'F1-Score']
    metric_values = [metrics['precision'], metrics['recall'], metrics['f1_score']]
    
    bars = ax.bar(metric_names, metric_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom')
    
    ax.grid(True, alpha=0.3)
    
    # 4. Timing breakdown
    ax = axes[1, 1]
    time_labels = ['Load', 'Cluster', 'Optimize']
    time_values = [
        metrics['load_time'],
        metrics['cluster_time'], 
        metrics['optimization_time']
    ]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    wedges, texts, autotexts = ax.pie(time_values, labels=time_labels, colors=colors,
                                     autopct='%1.1f%%', startangle=90)
    ax.set_title('Time Breakdown')
    
    plt.tight_layout()
    plt.savefig('ultrafast_yo_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Ultra-fast visualization saved: ultrafast_yo_results.png")


def export_ultra_fast_results(yo_framework):
    """Export results optimized for speed"""
    print("Exporting ultra-fast results...")
    
    if yo_framework.yo_candidates:
        # Export top results only for speed
        top_results = yo_framework.yo_candidates[:500]  # Limit export size
        
        results_data = []
        for rank, candidate in enumerate(top_results, 1):
            idx = candidate['index']
            row = yo_framework.df.iloc[idx]
            
            result_row = {
                'rank': rank,
                'original_index': idx,
                'final_score': candidate['final_score'],
                'match_score': candidate['match_score'],
                'diversity_score': candidate['diversity_score'],
                'clustering_score': candidate.get('clustering_score', 0),
                'is_ground_truth': idx in yo_framework.ground_truth_indices,
                'temperature': row[yo_framework.column_mapping['temperature']],
                'humidity': row[yo_framework.column_mapping['humidity']],
                'pressure': row[yo_framework.column_mapping['pressure']]
            }
            results_data.append(result_row)
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('ultrafast_yo_results.csv', index=False)
        print(f"Exported ultrafast_yo_results.csv ({len(results_df)} rows)")


def print_ultra_fast_summary(yo_framework, metrics):
    """Print optimized summary"""
    print("\n" + "="*70)
    print("ULTRA-FAST YO PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Original dataset: {len(yo_framework.df):,} rows")
    print(f"Processed sample: {metrics['samples_processed']:,} rows ({metrics['sampling_ratio']:.1%})")
    print(f"Total execution: {metrics['execution_time']:.2f}s")
    print(f"  - Data loading: {metrics['load_time']:.2f}s")
    print(f"  - Clustering: {metrics['cluster_time']:.2f}s") 
    print(f"  - Optimization: {metrics['optimization_time']:.2f}s")
    print(f"Throughput: {metrics['throughput']:.0f} records/second")
    
    if metrics['clustering_enabled']:
        print(f"Smart clustering: ENABLED ({metrics['clusters_found']} clusters)")
    else:
        print(f"Clustering: DISABLED")
    
    print()
    print("RESULTS:")
    print(f"  YO Candidates: {metrics['yo_candidates']:,}")
    print(f"  Ground Truth: {metrics['ground_truth']:,}")
    print(f"  Overlap: {metrics['overlap']:,}")
    print()
    print("PERFORMANCE:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print()
    print("TOP 10 CANDIDATES:")
    
    for i, candidate in enumerate(yo_framework.yo_candidates[:10]):
        idx = candidate['index']
        row = yo_framework.df.iloc[idx]
        is_gt = "✓" if idx in yo_framework.ground_truth_indices else "✗"
        
        temp = row[yo_framework.column_mapping['temperature']]
        humid = row[yo_framework.column_mapping['humidity']]
        press = row[yo_framework.column_mapping['pressure']]
        
        print(f"  {i+1:2d}. Row {idx:6d} | Score: {candidate['final_score']:.3f} | "
              f"T: {temp:6.2f} | H: {humid:.3f} | P: {press:7.2f} | GT: {is_gt}")
    
    print("="*70)


def run_ultra_fast_yo(csv_file, target_ranges, clustering_params=None, speed_params=None):
    """
    Complete Ultra-Fast YO pipeline - maximum speed optimization
    
    Args:
        csv_file (str): Path to CSV file
        target_ranges (dict): Target ranges for features
        clustering_params (dict): Clustering configuration (optional)
        speed_params (dict): Speed optimization parameters (optional)
    
    Returns:
        UltraFastYO framework instance
    """
    
    print("="*70)
    print("ULTRA-FAST YO - MAXIMUM SPEED OPTIMIZATION")
    print("="*70)
    
    try:
        # Initialize ultra-fast framework
        yo = UltraFastYO(csv_file, target_ranges, clustering_params, speed_params)
        
        # Find ground truth
        yo.find_ground_truth_fast()
        
        # Run ultra-fast optimization
        yo.run_ultra_fast_optimization()
        
        # Analyze performance
        metrics = yo.analyze_performance()
        
        # Create visualizations
        create_ultra_fast_visualizations(yo)
        
        # Export results
        export_ultra_fast_results(yo)
        
        # Print summary
        print_ultra_fast_summary(yo, metrics)
        
        print(f"\nUltra-Fast YO completed in {metrics['execution_time']:.2f} seconds!")
        return yo
        
    except Exception as e:
        print(f"Error in Ultra-Fast YO: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_synthetic_data_fast(filename, n_samples=50000, n_clusters=5, target_ranges=None):
    """
    Fast synthetic data generation for testing ultra-speed performance
    """
    print(f"Creating synthetic data (fast): {filename}")
    
    if target_ranges is None:
        target_ranges = {
            'temperature': [15, 20],
            'humidity': [0.3, 0.7],
            'pressure': [1013, 1017]
        }
    
    np.random.seed(42)
    
    # Fast cluster generation
    temp_range = target_ranges['temperature']
    humid_range = target_ranges['humidity']
    press_range = target_ranges['pressure']
    
    # Generate cluster centers quickly
    cluster_centers = []
    
    # Target clusters (in range)
    n_target = max(1, int(n_clusters * 0.6))
    for i in range(n_target):
        center = [
            np.random.uniform(temp_range[0], temp_range[1]),
            np.random.uniform(humid_range[0], humid_range[1]),
            np.random.uniform(press_range[0], press_range[1])
        ]
        cluster_centers.append(center)
    
    # Outlier clusters (outside range)
    for i in range(n_clusters - n_target):
        center = [
            np.mean(temp_range) + np.random.choice([-1, 1]) * np.random.uniform(3, 10),
            np.mean(humid_range) + np.random.choice([-1, 1]) * np.random.uniform(0.15, 0.4),
            np.mean(press_range) + np.random.choice([-1, 1]) * np.random.uniform(10, 25)
        ]
        cluster_centers.append(center)
    
    # Fast data generation
    all_data = []
    samples_per_cluster = n_samples // n_clusters
    
    for i, center in enumerate(cluster_centers):
        n_samples_cluster = samples_per_cluster
        if i == len(cluster_centers) - 1:
            n_samples_cluster = n_samples - len(all_data)
        
        # Variable cluster tightness
        cluster_std = np.random.uniform(0.8, 2.5)
        
        # Generate cluster data
        cluster_data = np.random.multivariate_normal(
            center, np.eye(3) * cluster_std, n_samples_cluster
        )
        all_data.extend(cluster_data)
    
    # Add noise (5% for speed)
    n_noise = int(n_samples * 0.05)
    for _ in range(n_noise):
        noise_point = [
            np.random.uniform(5, 35),
            np.random.uniform(0.05, 0.95),
            np.random.uniform(980, 1040)
        ]
        all_data.append(noise_point)
    
    # Convert to DataFrame
    all_data = np.array(all_data)
    df = pd.DataFrame(all_data, columns=[
        'ambient_temperature_celsius',
        'relative_humidity_fraction',
        'atmospheric_pressure_hpa'
    ])
    
    # Add minimal additional columns for realism
    df['sensor_id'] = np.random.randint(100, 200, len(df))
    df['timestamp'] = pd.date_range('2024-01-01', periods=len(df), freq='30s')
    
    # Clip values to realistic ranges
    df['relative_humidity_fraction'] = np.clip(df['relative_humidity_fraction'], 0, 1)
    
    # Save quickly
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} samples to {filename}")
    
    return df


# ============================================================================
# MAIN EXECUTION - ULTRA-FAST VERSION
# ============================================================================

if __name__ == "__main__":
    print("Ultra-Fast YO - Maximum Speed YO Framework")
    print("="*60)
    
    # Optimized target ranges
    target_ranges = {
        'temperature': [10, 30],    # °C  
        'humidity': [0.3, 0.7],     # decimal
        'pressure': [1013, 1017]    # hPa
    }
    
    # Speed-optimized clustering parameters
    clustering_params = {
        'enable_clustering': True,
        'sample_ratio': 0.98,           # Use only 3% for clustering
        'min_sample_size': 500,         # Smaller minimum
        'max_sample_size': 3000,        # Smaller maximum
        'cluster_radius': 0.2,          # Larger radius for faster clustering
        'min_cluster_size': 3,          # Smaller minimum cluster size
        'cluster_bias': 0.4,            # Moderate bias
        'max_clusters': 6,              # Fewer clusters to track
        'cluster_focus_prob': 0.75,     # Higher focus probability
        'use_pca': True,                # Enable PCA for speed
        'pca_variance': 0.90            # Lower variance requirement
    }
    
    # Ultra-speed parameters
    speed_params = {
        'max_data_size': 10000,          # Smaller working set
        'early_sample_ratio': 0.12,     # Smaller initial sample
        'chains': 30,                   # Fewer chains
        'max_iters': 5000,              # Fewer iterations
        'convergence_patience': 80,     # Less patience for speed
        'batch_size': 300,              # Smaller batches
        'vectorization_threshold': 500   # Lower threshold
    }
    
    print("\nUltra-Fast YO Features:")
    print("- Smart data sampling (uses only needed % of dataset)")
    print("- Intelligent clustering with PCA acceleration")
    print("- Adaptive neighborhood selection")
    print("- Early convergence detection")
    print("- Vectorized operations throughout")
    print("- Memory-optimized processing")
    print("- Fast synthetic data generation")
    
    print(f"\nSpeed optimizations:")
    print(f"- Max dataset size: {speed_params['max_data_size']:,}")
    print(f"- Clustering sample ratio: {clustering_params['sample_ratio']:.1%}")
    print(f"- Optimization chains: {speed_params['chains']}")
    print(f"- Max iterations per chain: {speed_params['max_iters']:,}")
    
    print("\nExample usage:")
    print("# Create fast synthetic data")
    print("create_synthetic_data_fast('fast_data.csv', n_samples=100000)")
    print()
    print("# Run Ultra-Fast YO")
    print("yo = run_ultra_fast_yo('fast_data.csv', target_ranges, clustering_params, speed_params)")
    print()
    print("# Or use existing data")
    print("yo = run_ultra_fast_yo('DATA.csv', target_ranges, clustering_params, speed_params)")
    
    print("\nReady for ultra-fast execution!")
    
    # Uncomment to run with synthetic data:
    # create_synthetic_data_fast('ultra_fast_data.csv', n_samples=100000)
    # yo = run_ultra_fast_yo('ultra_fast_data.csv', target_ranges, clustering_params, speed_params)
    
    # Uncomment to run with your existing data:
yo = run_ultra_fast_yo('DATA.csv', target_ranges, clustering_params, speed_params)