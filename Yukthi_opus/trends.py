import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class YukthiOpus:
    """
    Yukthi Opus: MCMC exploration + Greedy refinement + Simulated Annealing 
    controller with reheating and memory for climate trend analysis from CSV.
    """
    
    def __init__(self, chain_length=1000, T0=100.0, T_min=0.1, 
                 cooling_rate=0.95, reheat_threshold=50, memory_length=100,
                 local_jitter_std=0.1, global_jump_prob=0.2):
        """
        Initialize Yukthi Opus parameters.
        
        Parameters:
        -----------
        chain_length : int
            Number of MCMC iterations
        T0 : float
            Initial temperature for simulated annealing
        T_min : float
            Minimum temperature
        cooling_rate : float
            Temperature cooling factor
        reheat_threshold : int
            Steps without improvement before reheating
        memory_length : int
            Length of memory buffer for tracking best solutions
        local_jitter_std : float
            Standard deviation for local parameter jitter
        global_jump_prob : float
            Probability of making a global jump
        """
        self.chain_length = chain_length
        self.T0 = T0
        self.T_min = T_min
        self.cooling_rate = cooling_rate
        self.reheat_threshold = reheat_threshold
        self.memory_length = memory_length
        self.local_jitter_std = local_jitter_std
        self.global_jump_prob = global_jump_prob
        
        # Diagnostics
        self.diagnostics = {
            'evaluations': 0,
            'acceptances': 0,
            'rejections': 0,
            'local_moves': 0,
            'global_jumps': 0,
            'reheats': 0,
            'best_errors': [],
            'temperatures': []
        }
        
        # Memory for best solutions
        self.memory = []
        self.best_solution = None
        self.best_error = float('inf')
        
        # Data structure info
        self.data_info = {}
    
    def load_csv_data(self, csv_path, date_column=None, auto_detect=True):
        """
        Load and preprocess CSV climate data with intelligent column detection.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        date_column : str, optional
            Name of date column (auto-detected if None)
        auto_detect : bool
            Whether to auto-detect column types
        """
        print(f"📂 Loading data from: {csv_path}")
        
        # Load CSV
        try:
            data = pd.read_csv(csv_path)
            print(f"✅ Successfully loaded {len(data)} rows, {len(data.columns)} columns")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {e}")
        
        print(f"🔍 Columns found: {list(data.columns)}")
        
        # Auto-detect date column
        if date_column is None and auto_detect:
            date_candidates = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year']):
                    date_candidates.append(col)
            
            if date_candidates:
                date_column = date_candidates[0]
                print(f"🗓️ Auto-detected date column: {date_column}")
            else:
                print("⚠️ No date column detected, using row index as time")
        
        # Parse date column
        if date_column and date_column in data.columns:
            try:
                data['parsed_date'] = pd.to_datetime(data[date_column], infer_datetime_format=True)
                data['year'] = data['parsed_date'].dt.year
                data['month'] = data['parsed_date'].dt.month
                data['day_of_year'] = data['parsed_date'].dt.dayofyear
                print(f"✅ Date parsing successful: {data['parsed_date'].min()} to {data['parsed_date'].max()}")
            except:
                print("⚠️ Date parsing failed, using row index")
                data['parsed_date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
                data['year'] = data['parsed_date'].dt.year
                data['month'] = data['parsed_date'].dt.month
                data['day_of_year'] = data['parsed_date'].dt.dayofyear
        else:
            # Create artificial date index
            data['parsed_date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
            data['year'] = data['parsed_date'].dt.year
            data['month'] = data['parsed_date'].dt.month
            data['day_of_year'] = data['parsed_date'].dt.dayofyear
        
        # Auto-detect climate variables
        climate_variables = []
        
        # Common climate variable patterns
        temp_patterns = ['temp', 'temperature', 'celsius', 'fahrenheit', 'degree']
        humidity_patterns = ['humid', 'moisture', 'rh', 'relative_humidity']
        pressure_patterns = ['press', 'pressure', 'hpa', 'mbar', 'atm', 'pascal']
        
        all_patterns = {
            'temperature': temp_patterns,
            'humidity': humidity_patterns,  
            'pressure': pressure_patterns
        }
        
        detected_vars = {}
        
        for col in data.columns:
            if col in ['parsed_date', 'year', 'month', 'day_of_year', date_column]:
                continue
                
            col_lower = col.lower()
            
            # Check if it's numeric
            if not pd.api.types.is_numeric_dtype(data[col]):
                continue
                
            # Pattern matching
            for var_type, patterns in all_patterns.items():
                if any(pattern in col_lower for pattern in patterns):
                    detected_vars[var_type] = col
                    break
            else:
                # If no pattern match, consider as generic climate variable
                if col not in detected_vars.values():
                    detected_vars[f'variable_{len(detected_vars)+1}'] = col
        
        print(f"🌡️ Detected climate variables: {detected_vars}")
        
        # Store data info
        self.data_info = {
            'original_columns': list(data.columns),
            'detected_variables': detected_vars,
            'date_column': date_column,
            'data_shape': data.shape,
            'date_range': (data['parsed_date'].min(), data['parsed_date'].max())
        }
        
        return data, detected_vars
    
    def trendline_model(self, params, x):
        """
        Flexible trendline model (polynomial + seasonality).
        params: [poly_coeffs..., seasonal_amplitude, seasonal_phase]
        """
        if len(params) < 3:
            # Simple linear model
            return params[0] * x + params[1] if len(params) >= 2 else params[0] * np.ones_like(x)
        
        poly_order = len(params) - 2
        poly_coeffs = params[:poly_order]
        seasonal_amp = params[-2]
        seasonal_phase = params[-1]
        
        # Polynomial trend
        trend = np.polyval(poly_coeffs, x)
        
        # Seasonal component (assuming yearly cycle)
        seasonal = seasonal_amp * np.sin(2 * np.pi * x / 365 + seasonal_phase)
        
        return trend + seasonal
    
    def evaluate_trendline(self, params, x, y, weights=None):
        """Evaluate trendline quality (lower is better)."""
        self.diagnostics['evaluations'] += 1
        
        try:
            predicted = self.trendline_model(params, x)
            residuals = y - predicted
            
            if weights is not None:
                mse = np.mean(weights * residuals**2)
            else:
                mse = np.mean(residuals**2)
            
            # Add regularization to prevent overfitting
            regularization = 0.001 * np.sum(np.array(params)**2)
            
            return mse + regularization
            
        except:
            return 1e10  # Return high error for invalid parameters
    
    def propose_move(self, current_params, temperature):
        """Propose a new parameter set using local jitter or global jump."""
        if np.random.random() < self.global_jump_prob:
            # Global jump - sample from wider distribution
            self.diagnostics['global_jumps'] += 1
            new_params = np.random.normal(0, 0.5, len(current_params))
        else:
            # Local jitter - small perturbation
            self.diagnostics['local_moves'] += 1
            jitter_scale = self.local_jitter_std * np.sqrt(temperature / self.T0)
            jitter = np.random.normal(0, jitter_scale, len(current_params))
            new_params = current_params + jitter
        
        return new_params
    
    def accept_move(self, current_error, new_error, temperature):
        """Decide whether to accept a proposed move using simulated annealing."""
        if new_error < current_error:
            return True
        else:
            # Accept with probability based on temperature
            delta = new_error - current_error
            prob = np.exp(-delta / max(temperature, 1e-10))
            return np.random.random() < prob
    
    def greedy_refinement(self, params, x, y, weights=None, iterations=10):
        """Local greedy refinement of parameters."""
        best_params = params.copy()
        best_error = self.evaluate_trendline(best_params, x, y, weights)
        
        for _ in range(iterations):
            # Try small improvements in each parameter
            for i in range(len(params)):
                for delta in [-0.01, 0.01]:
                    test_params = best_params.copy()
                    test_params[i] += delta
                    
                    error = self.evaluate_trendline(test_params, x, y, weights)
                    if error < best_error:
                        best_params = test_params.copy()
                        best_error = error
        
        return best_params, best_error
    
    def update_memory(self, params, error):
        """Update memory buffer with best solutions."""
        self.memory.append((params.copy(), error))
        
        # Keep only best solutions in memory
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_length:
            self.memory = self.memory[:self.memory_length]
        
        # Update global best
        if error < self.best_error:
            self.best_solution = params.copy()
            self.best_error = error
    
    def explore_trends(self, data, variable_col, weights=None):
        """
        Main MCMC exploration with simulated annealing and greedy refinement.
        """
        print(f"🔍 Exploring trends for {variable_col}...")
        
        # Prepare data
        x = data['day_of_year'].values
        y = data[variable_col].values
        
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        if len(x) < 10:
            print(f"⚠️ Insufficient data for {variable_col}")
            return None
        
        if weights is None:
            weights = np.ones(len(x))
        else:
            weights = weights[mask]
        
        # Initialize parameters based on data characteristics
        y_std = np.std(y)
        n_params = 5  # [linear, quadratic, intercept, seasonal_amp, seasonal_phase]
        
        # Smart initialization
        current_params = np.array([
            0.0,  # linear trend
            0.0,  # quadratic trend  
            np.mean(y),  # intercept
            y_std * 0.5,  # seasonal amplitude
            0.0   # seasonal phase
        ])
        
        current_error = self.evaluate_trendline(current_params, x, y, weights)
        
        # Initialize temperature
        temperature = self.T0
        steps_without_improvement = 0
        
        # MCMC chain
        chain_params = []
        chain_errors = []
        
        for step in range(self.chain_length):
            # Propose new parameters
            new_params = self.propose_move(current_params, temperature)
            new_error = self.evaluate_trendline(new_params, x, y, weights)
            
            # Accept/reject move
            if self.accept_move(current_error, new_error, temperature):
                current_params = new_params
                current_error = new_error
                self.diagnostics['acceptances'] += 1
                
                if new_error < self.best_error:
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
            else:
                self.diagnostics['rejections'] += 1
                steps_without_improvement += 1
            
            # Store chain state
            chain_params.append(current_params.copy())
            chain_errors.append(current_error)
            
            # Update memory
            self.update_memory(current_params, current_error)
            
            # Cooling
            temperature = max(temperature * self.cooling_rate, self.T_min)
            
            # Reheating if stuck
            if steps_without_improvement >= self.reheat_threshold:
                temperature = self.T0 * 0.3  # Partial reheat
                self.diagnostics['reheats'] += 1
                steps_without_improvement = 0
                
                # Jump to a good solution from memory
                if self.memory:
                    current_params = self.memory[0][0].copy()  # Best from memory
                    current_error = self.memory[0][1]
            
            # Store diagnostics
            self.diagnostics['best_errors'].append(self.best_error)
            self.diagnostics['temperatures'].append(temperature)
            
            # Progress update
            if (step + 1) % max(1, self.chain_length // 10) == 0:
                progress = (step + 1) / self.chain_length * 100
                print(f"  Progress: {progress:.0f}% - Best Error: {self.best_error:.6f}")
        
        # Final greedy refinement
        print("  🎯 Final greedy refinement...")
        refined_params, refined_error = self.greedy_refinement(
            self.best_solution, x, y, weights)
        
        if refined_error < self.best_error:
            self.best_solution = refined_params
            self.best_error = refined_error
        
        return {
            'best_params': self.best_solution,
            'best_error': self.best_error,
            'chain_params': chain_params,
            'chain_errors': chain_errors,
            'final_trend': self.trendline_model(self.best_solution, x),
            'x_data': x,
            'y_data': y
        }
    
    def analyze_csv_data(self, csv_path, date_column=None, variable_weights=None, 
                         tolerance_threshold=0.1, custom_variables=None):
        """
        Complete analysis of CSV climate data.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file
        date_column : str, optional
            Name of date column
        variable_weights : dict, optional
            Weights for different variables
        tolerance_threshold : float
            Smoothing tolerance
        custom_variables : list, optional
            Manually specify which columns to analyze
        """
        print("🌡️ Starting Yukthi Opus CSV Climate Analysis...")
        print("=" * 60)
        
        # Load and preprocess data
        data, detected_vars = self.load_csv_data(csv_path, date_column)
        
        # Use custom variables if provided
        if custom_variables:
            analysis_vars = {}
            for var in custom_variables:
                if var in data.columns:
                    analysis_vars[var] = var
                else:
                    print(f"⚠️ Variable '{var}' not found in data")
        else:
            analysis_vars = detected_vars
        
        if not analysis_vars:
            raise ValueError("No valid variables found for analysis")
        
        print(f"📊 Analyzing variables: {list(analysis_vars.keys())}")
        
        results = {}
        
        for var_name, col_name in analysis_vars.items():
            print(f"\n{'='*40}")
            print(f"Analyzing: {var_name} ({col_name})")
            print(f"{'='*40}")
            
            # Reset diagnostics for each variable
            self.diagnostics = {k: 0 if isinstance(v, (int, float)) else [] 
                                for k, v in self.diagnostics.items()}
            self.memory = []
            self.best_solution = None
            self.best_error = float('inf')
            
            # Explore trends
            var_results = self.explore_trends(data, col_name, variable_weights)
            
            if var_results is None:
                continue
            
            # Monthly aggregation
            monthly_data = data.groupby('month')[col_name].agg(['mean', 'std', 'count']).reset_index()
            monthly_data['std'] = monthly_data['std'].fillna(0)
            
            # Yearly aggregation  
            yearly_data = data.groupby('year')[col_name].agg(['mean', 'std', 'count']).reset_index()
            yearly_data['std'] = yearly_data['std'].fillna(0)
            
            results[var_name] = {
                'column_name': col_name,
                'trend_results': var_results,
                'monthly_stats': monthly_data,
                'yearly_stats': yearly_data,
                'diagnostics': self.diagnostics.copy(),
                'data_stats': {
                    'mean': data[col_name].mean(),
                    'std': data[col_name].std(),
                    'min': data[col_name].min(),
                    'max': data[col_name].max(),
                    'count': data[col_name].count()
                }
            }
        
        # Generate narrative summary
        narrative = self.generate_narrative_summary(results, data)
        
        return results, narrative, data
    
    def generate_narrative_summary(self, results, data):
        """Generate natural language summary of findings."""
        narrative = []
        
        narrative.append("🔍 YUKTHI OPUS CSV CLIMATE ANALYSIS SUMMARY")
        narrative.append("=" * 60)
        
        # Dataset overview
        narrative.append(f"\n📊 DATASET OVERVIEW:")
        narrative.append(f"    • Total records: {len(data):,}")
        narrative.append(f"    • Date range: {self.data_info['date_range'][0].strftime('%Y-%m-%d')} to {self.data_info['date_range'][1].strftime('%Y-%m-%d')}")
        narrative.append(f"    • Variables analyzed: {len(results)}")
        
        # Individual variable analysis
        for var_name, var_data in results.items():
            col_name = var_data['column_name']
            monthly_stats = var_data['monthly_stats']
            yearly_stats = var_data['yearly_stats']
            stats = var_data['data_stats']
            
            # Find peak and low months
            if len(monthly_stats) > 0:
                peak_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'month']
                low_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'month']
                
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                peak_month_name = month_names[int(peak_month)] if peak_month <= 12 else 'Unknown'
                low_month_name = month_names[int(low_month)] if low_month <= 12 else 'Unknown'
            else:
                peak_month_name = low_month_name = "Unknown"
            
            # Yearly trend analysis
            if len(yearly_stats) > 1:
                yearly_trend = np.polyfit(yearly_stats['year'], yearly_stats['mean'], 1)[0]
                trend_direction = ("increasing" if yearly_trend > 0.01 else 
                                   "decreasing" if yearly_trend < -0.01 else "stable")
            else:
                yearly_trend = 0
                trend_direction = "stable"
            
            narrative.append(f"\n📈 {var_name.upper()} ({col_name}):")
            narrative.append(f"    • Data range: {stats['min']:.2f} to {stats['max']:.2f} (mean: {stats['mean']:.2f})")
            narrative.append(f"    • Seasonal pattern: Peaks in {peak_month_name}, lowest in {low_month_name}")
            narrative.append(f"    • Long-term trend: {trend_direction} ({yearly_trend:.4f} units/year)")
            narrative.append(f"    • Model fit error: {var_data['trend_results']['best_error']:.6f}")
            narrative.append(f"    • Data completeness: {stats['count']}/{len(data)} records ({stats['count']/len(data)*100:.1f}%)")
            
            # Diagnostics
            diag = var_data['diagnostics']
            total_moves = diag['acceptances'] + diag['rejections']
            if total_moves > 0:
                acceptance_rate = diag['acceptances'] / total_moves * 100
                narrative.append(f"    • MCMC performance: {acceptance_rate:.1f}% acceptance, {diag['reheats']} reheats")
                narrative.append(f"    • Search strategy: {diag['global_jumps']} global jumps, {diag['local_moves']} local moves")
        
        # Overall insights
        narrative.append(f"\n🎯 OVERALL INSIGHTS:")
        
        if len(results) >= 2:
            # Cross-variable correlations
            var_names = list(results.keys())
            correlations = []
            
            for i, var1 in enumerate(var_names):
                for var2 in var_names[i+1:]:
                    col1 = results[var1]['column_name']
                    col2 = results[var2]['column_name']
                    
                    # Calculate correlation
                    mask = ~(data[col1].isna() | data[col2].isna())
                    if mask.sum() > 10:
                        corr = data.loc[mask, col1].corr(data.loc[mask, col2])
                        correlations.append((var1, var2, corr))
            
            if correlations:
                strong_corrs = [(v1, v2, c) for v1, v2, c in correlations if abs(c) > 0.5]
                if strong_corrs:
                    narrative.append("    • Strong correlations detected:")
                    for var1, var2, corr in strong_corrs:
                        corr_type = "positive" if corr > 0 else "negative"
                        narrative.append(f"       - {var1} ↔ {var2}: {corr_type} correlation (r={corr:.3f})")
        
        # Data quality assessment
        total_records = len(data)
        complete_records = len(data.dropna())
        completeness = complete_records / total_records * 100
        
        narrative.append(f"    • Data quality: {completeness:.1f}% complete records")
        
        if completeness < 80:
            narrative.append("    ⚠️ Consider data cleaning for improved analysis")
        elif completeness > 95:
            narrative.append("    ✅ Excellent data quality detected")
        
        narrative.append(f"\n✅ Analysis completed successfully with {self.diagnostics.get('evaluations', 0)} model evaluations!")
        
        return "\n".join(narrative)
    
    def plot_csv_results(self, results, data, figsize=(16, 12)):
        """Create comprehensive visualization of CSV results."""
        n_vars = len(results)
        if n_vars == 0:
            return None
            
        fig, axes = plt.subplots(n_vars, 3, figsize=figsize)
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_vars))
        
        for i, (var_name, var_results) in enumerate(results.items()):
            col_name = var_results['column_name']
            color = colors[i]
            
            # Original data vs trend
            ax1 = axes[i, 0]
            trend_res = var_results['trend_results']
            x = trend_res['x_data']
            y = trend_res['y_data']
            trend = trend_res['final_trend']
            
            ax1.scatter(x, y, alpha=0.4, s=2, color=color, label='Data', rasterized=True)
            
            # Sort for smooth trend line
            sort_idx = np.argsort(x)
            ax1.plot(x[sort_idx], trend[sort_idx], color='red', linewidth=2, label='YO Trend')
            
            ax1.set_xlabel('Day of Year')
            ax1.set_ylabel(f'{var_name}')
            ax1.set_title(f'{var_name} - Trend Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Monthly patterns
            ax2 = axes[i, 1]
            monthly = var_results['monthly_stats']
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if len(monthly) > 0:
                ax2.errorbar(monthly['month'], monthly['mean'], 
                             yerr=monthly['std'], marker='o', 
                             color=color, capsize=5, linewidth=2, markersize=6)
                ax2.set_xticks(range(1, 13))
                ax2.set_xticklabels(months, rotation=45)
            
            ax2.set_ylabel(f'{var_name}')
            ax2.set_title(f'{var_name} - Monthly Patterns')
            ax2.grid(True, alpha=0.3)
            
            # Yearly trends
            ax3 = axes[i, 2]
            yearly = var_results['yearly_stats']
            
            if len(yearly) > 0:
                ax3.errorbar(yearly['year'], yearly['mean'], 
                             yerr=yearly['std'], marker='s', 
                             color=color, capsize=5, linewidth=2, markersize=6)
                
                # Add trend line if multiple years
                if len(yearly) > 1:
                    z = np.polyfit(yearly['year'], yearly['mean'], 1)
                    trend_line = np.polyval(z, yearly['year'])
                    ax3.plot(yearly['year'], trend_line, '--', color='gray', alpha=0.7)
            
            ax3.set_xlabel('Year')
            ax3.set_ylabel(f'{var_name}')
            ax3.set_title(f'{var_name} - Yearly Trends')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, results, narrative, output_path="yukthi_opus_results"):
        """Save results to files."""
        print(f"💾 Saving results to {output_path}...")
        
        # Save narrative
        with open(f"{output_path}_summary.txt", "w") as f:
            f.write(narrative)
        
        # Save detailed results as CSV
        summary_data = []
        for var_name, var_data in results.items():
            stats = var_data['data_stats']
            diag = var_data['diagnostics']
            
            summary_data.append({
                'variable': var_name,
                'column_name': var_data['column_name'],
                'mean': stats['mean'],
                'std': stats['std'],
                'min_value': stats['min'],
                'max_value': stats['max'],
                'data_count': stats['count'],
                'model_error': var_data['trend_results']['best_error'],
                'mcmc_acceptances': diag['acceptances'],
                'mcmc_rejections': diag['rejections'],
                'global_jumps': diag['global_jumps'],
                'local_moves': diag['local_moves'],
                'reheats': diag['reheats']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_path}_summary.csv", index=False)
        
        print(f"✅ Results saved!")
        return summary_df

# Convenience function to analyze climate CSV data
def analyze_climate_csv(csv_path, **kwargs):
    """
    Convenience function to analyze climate CSV data.
    
    Parameters:
    -----------
    csv_path : str
        Path to CSV file
    **kwargs : 
        Additional parameters for YukthiOpus initialization
    
    Returns:
    --------
    results, narrative, data, yo_instance
    """
    print("🚀 Starting Yukthi Opus CSV Analysis...")
    
    # Initialize YO system
    yo = YukthiOpus(**kwargs)
    
    # Run analysis
    results, narrative, data = yo.analyze_csv_data(csv_path)
    
    # Display results
    print("\n" + narrative)
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    fig = yo.plot_csv_results(results, data)
    if fig:
        plt.show()
    
    return results, narrative, data, yo

# Example usage when you have a CSV file
if __name__ == "__main__":
    # Example usage - replace 'your_climate_data.csv' with your actual file path
    csv_file = "DATA.csv"  # <-- PUT YOUR CSV PATH HERE
    
    # Option 1: Quick analysis with defaults
    try:
        results, narrative, data, yo_instance = analyze_climate_csv(
            csv_file,
            chain_length=500,  # Adjust based on your data size
            T0=50.0,
            cooling_rate=0.98
        )
        
        # Save the results after a successful analysis
        yo_instance.save_results(results, narrative, output_path="climate_analysis_results")

    except FileNotFoundError:
        print(f"\n❌ ERROR: The file '{csv_file}' was not found.")
        print("Please make sure the CSV file exists in the correct directory or provide the full path.")
        
        # As an example, create a dummy dataset for demonstration
        print("\n🛠️ Creating a dummy 'dummy_climate_data.csv' for demonstration purposes...")
        dummy_dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=365*3, freq='D'))
        dummy_df = pd.DataFrame({
            'timestamp': dummy_dates,
            'avg_temp_celsius': 15 + 10 * np.sin(2 * np.pi * dummy_dates.dayofyear / 365) + 
                                0.2 * (dummy_dates.year - 2020) + np.random.normal(0, 2.5, len(dummy_dates)),
            'relative_humidity': 60 - 20 * np.sin(2 * np.pi * dummy_dates.dayofyear / 365 + np.pi/2) + 
                                 np.random.normal(0, 5, len(dummy_dates))
        })
        dummy_csv_path = "dummy_climate_data.csv"
        dummy_df.to_csv(dummy_csv_path, index=False)
        print(f"✅ Dummy data saved to '{dummy_csv_path}'. You can re-run the script with this filename.")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during the analysis: {e}")

    finally:
        print("\n🏁 Yukthi Opus analysis script finished.")