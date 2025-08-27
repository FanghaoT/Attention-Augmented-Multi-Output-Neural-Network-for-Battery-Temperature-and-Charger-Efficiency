import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

inputfolder = 'InputData'
outputfolder = 'OutputData'

print("Script started")
print("Input folder:", inputfolder)
print("Output folder:", outputfolder)
print("Current working directory:", os.getcwd())


# Section 1: 
# Load the raw data of charging data
raw_data = pd.read_csv(os.path.join(inputfolder, '7charging_data_group_outliers_removed.csv'))
#find the column index of the data
print("Raw data columns:", raw_data.columns.tolist())
print("Raw data shape:", raw_data.shape)
print("Raw data head:")
print(raw_data.head())

#delete the columns that are "Unnamed"
raw_data = raw_data.drop(raw_data.columns[raw_data.columns.str.contains('unnamed',case = False)],axis = 1)
#find the column index of the data
print("After cleaning - Raw data columns:", raw_data.columns.tolist())
print(raw_data.head())

#create a new dataframe to store the processed data
# Select relevant columns for analysis (keep all 5 variables but sample based on 3 inputs)
relevant_columns = ['SOC', 'ChargingPower_kW', 'AmbientTemp', 'ChargingEfficiency', 'BatteryTemp']

standard_data = raw_data[relevant_columns].copy()

print("Original data shape:", standard_data.shape)
print("Standard data head:")
print(standard_data.head())

# Implement Grid-based Data Balancing for Uniform Distribution (3D version)
def create_uniform_dataset_3d(data, target_size=20000, n_bins=[10, 12, 8]):
    """
    Create a uniform distribution in 3D input space using grid-based sampling
    Consider SOC, ChargingPower_kW, AmbientTemp for sampling
    1. For data-rich blocks (current_count >= target_per_bin): 
       - Randomly sample target_per_bin samples without replacement
    2. For data-poor blocks (current_count < target_per_bin):
       - Copy the entire group multiple times to fill up to target_per_bin samples
       - Add small noise to each copy to create variation
       - If we exceed target_per_bin, randomly sample to get exactly target_per_bin
    
    Parameters:
    - data: DataFrame with all 5 variables
    - target_size: desired total number of samples
    - n_bins: number of bins for each dimension [SOC, ChargingPower_kW, AmbientTemp]
    """
    print("Creating uniform dataset distribution in 3D space (SOC, Power, AmbientTemp)...")
    
    # Define input columns for sampling (3 variables only)
    input_cols = ['SOC', 'ChargingPower_kW', 'AmbientTemp']
    
    # Create bins for each dimension
    soc_bins = pd.cut(data['SOC'], bins=n_bins[0], labels=False, include_lowest=True)
    power_bins = pd.cut(data['ChargingPower_kW'], bins=n_bins[1], labels=False, include_lowest=True)
    ambient_temp_bins = pd.cut(data['AmbientTemp'], bins=n_bins[2], labels=False, include_lowest=True)
    
    # Add bin information to data
    data_with_bins = data.copy()
    data_with_bins['soc_bin'] = soc_bins
    data_with_bins['power_bin'] = power_bins
    data_with_bins['ambient_temp_bin'] = ambient_temp_bins
    
    # Calculate total number of possible bins
    total_bins = n_bins[0] * n_bins[1] * n_bins[2]
    target_per_bin = target_size // total_bins
    
    print(f"Total possible bins: {total_bins}")
    print(f"Target samples per bin: {target_per_bin}")
    
    # Group by bins and analyze distribution
    grouped = data_with_bins.groupby(['soc_bin', 'power_bin', 'ambient_temp_bin'])
    bin_stats = []
    
    for name, group in grouped:
        bin_stats.append({
            'soc_bin': name[0],
            'power_bin': name[1], 
            'ambient_temp_bin': name[2],
            'count': len(group),
            'group_data': group
        })
    
    print(f"Number of non-empty bins: {len(bin_stats)}")
    
    # Analyze distribution
    counts = [stat['count'] for stat in bin_stats]
    print(f"Bin count statistics:")
    print(f"  Min: {min(counts)}, Max: {max(counts)}")
    print(f"  Mean: {np.mean(counts):.2f}, Std: {np.std(counts):.2f}")
    
    # Create balanced dataset
    balanced_data = []
    
    for stat in bin_stats:
        group = stat['group_data']
        current_count = stat['count']
        
        if current_count >= target_per_bin:
            # For data-rich blocks: randomly select target_per_bin samples without replacement
            # This ensures no duplicates within the same block
            sampled = group.sample(n=target_per_bin, random_state=42)
            balanced_data.append(sampled)
        else:
            # For data-poor blocks: copy existing samples to fill up to target_per_bin
            # Calculate how many copies we need to reach target_per_bin
            n_needed = target_per_bin - current_count
            
            if n_needed > 0:
                # Calculate how many times we need to copy the entire group
                copies_needed = (n_needed + current_count - 1) // current_count  # Ceiling division
                
                # Add the original group
                balanced_data.append(group)
                
                # Add copies of the group to reach target_per_bin
                for copy_idx in range(copies_needed):
                    # Create a copy of the group with small noise to avoid exact duplicates
                    group_copy = group.copy()
                    
                    # Add very small noise to all features to create variation
                    soc_noise = np.random.normal(0, 0.00001)  # Very small noise for SOC
                    power_noise = np.random.normal(0, 0.00001)  # Very small noise for power
                    ambient_temp_noise = np.random.normal(0, 0.00001)  # Very small noise for ambient temperature
                    efficiency_noise = np.random.normal(0, 0.00001)  # Very small noise for efficiency
                    battery_temp_noise = np.random.normal(0, 0.00001)  # Small noise for battery temperature
                    
                    group_copy['SOC'] += soc_noise
                    group_copy['ChargingPower_kW'] += power_noise
                    group_copy['AmbientTemp'] += ambient_temp_noise
                    group_copy['ChargingEfficiency'] += efficiency_noise
                    group_copy['BatteryTemp'] += battery_temp_noise
                    
                    # Ensure values stay within reasonable bounds
                    group_copy['SOC'] = np.clip(group_copy['SOC'], 
                                              data['SOC'].min(), data['SOC'].max())
                    group_copy['ChargingPower_kW'] = np.clip(group_copy['ChargingPower_kW'],
                                                           data['ChargingPower_kW'].min(), data['ChargingPower_kW'].max())
                    group_copy['AmbientTemp'] = np.clip(group_copy['AmbientTemp'],
                                                      data['AmbientTemp'].min(), data['AmbientTemp'].max())
                    group_copy['ChargingEfficiency'] = np.clip(group_copy['ChargingEfficiency'],
                                                             data['ChargingEfficiency'].min(), data['ChargingEfficiency'].max())
                    group_copy['BatteryTemp'] = np.clip(group_copy['BatteryTemp'],
                                                      data['BatteryTemp'].min(), data['BatteryTemp'].max())
                    
                    balanced_data.append(group_copy)
                
                # If we have more samples than needed, randomly sample to get exactly target_per_bin
                total_samples_added = current_count + (copies_needed * current_count)
                if total_samples_added > target_per_bin:
                    # Combine all samples for this bin
                    all_bin_samples = pd.concat(balanced_data[-copies_needed-1:], ignore_index=True)
                    # Randomly sample exactly target_per_bin samples
                    final_samples = all_bin_samples.sample(n=target_per_bin, random_state=42)
                    # Replace the last few entries with the final samples
                    balanced_data = balanced_data[:-copies_needed-1]
                    balanced_data.append(final_samples)
            else:
                # If we have enough samples, just add the original group
                balanced_data.append(group)
    
    # Combine all balanced data
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    
    # Remove bin columns
    balanced_df = balanced_df.drop(['soc_bin', 'power_bin', 'ambient_temp_bin'], axis=1)
    
    print(f"Original dataset size: {len(data)}")
    print(f"Balanced dataset size: {len(balanced_df)}")
    
    return balanced_df




# Apply uniform distribution balancing
print("Original data ranges:")
print(f"SOC: {standard_data['SOC'].min():.3f} to {standard_data['SOC'].max():.3f}")
print(f"ChargingPower_kW: {standard_data['ChargingPower_kW'].min():.3f} to {standard_data['ChargingPower_kW'].max():.3f}")
print(f"AmbientTemp: {standard_data['AmbientTemp'].min():.3f} to {standard_data['AmbientTemp'].max():.3f}")
print(f"BatteryTemp: {standard_data['BatteryTemp'].min():.3f} to {standard_data['BatteryTemp'].max():.3f}")
print(f"ChargingEfficiency: {standard_data['ChargingEfficiency'].min():.3f} to {standard_data['ChargingEfficiency'].max():.3f}")

# Use 3D balancing based on SOC, ChargingPower_kW, AmbientTemp
print("\nUsing 3D data balancing (SOC, ChargingPower_kW, AmbientTemp)")
standard_data_unique = create_uniform_dataset_3d(standard_data, target_size=20000, n_bins=[10, 8, 12])

# Visualize distribution improvement
def visualize_distribution_comparison(original_data, balanced_data):
    """Compare distribution before and after balancing"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    input_cols = ['SOC', 'ChargingPower_kW', 'AmbientTemp']
    
    for i, col in enumerate(input_cols):
        # Original distribution
        axes[0, i].hist(original_data[col], bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[0, i].set_title(f'Original {col} Distribution')
        axes[0, i].set_xlabel(col)
        axes[0, i].set_ylabel('Frequency')
        
        # Balanced distribution
        axes[1, i].hist(balanced_data[col], bins=30, alpha=0.7, color='blue', edgecolor='black')
        axes[1, i].set_title(f'Balanced {col} Distribution')
        axes[1, i].set_xlabel(col)
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outputfolder, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    print("Distribution comparison plot saved!")
    
    # Print distribution statistics
    print("\nDistribution Statistics Comparison:")
    for col in input_cols:
        print(f"\n{col}:")
        print(f"  Original - Mean: {original_data[col].mean():.3f}, Std: {original_data[col].std():.3f}")
        print(f"  Balanced - Mean: {balanced_data[col].mean():.3f}, Std: {balanced_data[col].std():.3f}")

# Visualize the distribution improvement
visualize_distribution_comparison(standard_data, standard_data_unique)


standard_data_unique.to_csv(os.path.join(outputfolder,'Standard_data_unique.csv'),index=False)

