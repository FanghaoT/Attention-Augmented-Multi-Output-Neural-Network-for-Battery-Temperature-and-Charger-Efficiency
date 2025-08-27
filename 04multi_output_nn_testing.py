"""
Multi-Output Neural Network Testing Program

This script loads the trained multi-output neural network model and tests it on the full dataset.
The results are saved with the same format as the training program but with '04' prefix.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import json
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import pickle
from sklearn.preprocessing import StandardScaler

class MultiOutputNet(nn.Module):
    """
    改进的多输出神经网络 - 温度替换架构
    根据领域知识设计：
    - 电池温度预测：基于原始输入 [SOC, ChargingPower, AmbientTemp]
    - 效率预测：使用预测的电池温度替换环境温度 [SOC, ChargingPower, PredictedBatteryTemp]
    
    关键创新：效率预测使用预测的电池温度而非环境温度，更符合物理规律
    """
    
    def __init__(self, input_size=3, shared_layer_sizes=None, temp_layer_sizes=None, eff_layer_sizes=None):
        super(MultiOutputNet, self).__init__()
        
        # 共享特征提取层
        shared_layers = []
        in_features = input_size
        for i, out_features in enumerate(shared_layer_sizes):
            shared_layers.append(nn.Linear(in_features, out_features))
            shared_layers.append(nn.ReLU())
            in_features = out_features
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 电池温度预测分支
        temp_layers = []
        temp_in_features = shared_layer_sizes[-1]  # 来自共享层的输出
        for i, out_features in enumerate(temp_layer_sizes):
            temp_layers.append(nn.Linear(temp_in_features, out_features))
            if i < len(temp_layer_sizes) - 1:  # 不是最后一层
                temp_layers.append(nn.ReLU())
            temp_in_features = out_features
        
        self.battery_temp_branch = nn.Sequential(*temp_layers)
        
        # 效率预测分支 (使用预测的电池温度替换环境温度)
        eff_layers = []
        eff_in_features = shared_layer_sizes[-1]  # 共享特征，不包含额外输入
        for i, out_features in enumerate(eff_layer_sizes):
            eff_layers.append(nn.Linear(eff_in_features, out_features))
            if i < len(eff_layer_sizes) - 1:  # 不是最后一层
                eff_layers.append(nn.ReLU())
            eff_in_features = out_features
        
        self.efficiency_branch = nn.Sequential(*eff_layers)
    
    def forward(self, x):
        # x: [SOC, ChargingPower_kW, AmbientTemp]
        # 使用温度分支的共享特征提取
        temp_shared_features = self.shared_layers(x)
        
        # 预测电池温度
        battery_temp = self.battery_temp_branch(temp_shared_features)
        
        # 创建用于效率预测的修改输入：用预测的电池温度替换环境温度
        # 原始输入: [SOC, ChargingPower, AmbientTemp]
        # 修改后: [SOC, ChargingPower, PredictedBatteryTemp]
        modified_input = torch.cat([x[:, :2], battery_temp], dim=1)  # 保留SOC和Power，用预测温度替换环境温度
        
        # 使用修改后的输入提取共享特征（使用相同的共享层）
        eff_shared_features = self.shared_layers(modified_input)
        
        # 预测效率
        efficiency = self.efficiency_branch(eff_shared_features)
        
        return battery_temp, efficiency

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataset():
    # Load data from Standard_data_unique.csv (raw data)
    data_path = os.path.join('OutputData', 'Standard_data_unique.csv')
    DATA = pd.read_csv(data_path)
    
    # Extract input features: SOC, ChargingPower_kW, AmbientTemp
    soc = DATA.loc[:, 'SOC'].values.reshape(-1, 1)
    charging_power = DATA.loc[:, 'ChargingPower_kW'].values.reshape(-1, 1)
    ambient_temp = DATA.loc[:, 'AmbientTemp'].values.reshape(-1, 1)
    
    # Extract outputs: BatteryTemp, ChargingEfficiency
    battery_temp = DATA.loc[:, 'BatteryTemp'].values.reshape(-1, 1)
    eff = DATA.loc[:, 'ChargingEfficiency'].values.reshape(-1, 1)

    # Prepare input and output tensors
    temp_input = np.concatenate((soc, charging_power, ambient_temp), axis=1)
    temp_output = np.concatenate((battery_temp, eff), axis=1)
    
    # Load scalers from training instead of fitting new ones
    try:
        with open('scaler_multi_output_input.pkl', 'rb') as f:
            input_scaler = pickle.load(f)
        with open('scaler_multi_output_output.pkl', 'rb') as f:
            output_scaler = pickle.load(f)
        
        # Transform the data using loaded scalers
        temp_input_scaled = input_scaler.transform(temp_input)
        temp_output_scaled = output_scaler.transform(temp_output)
        print("Loaded scalers from training and transformed data")
    except FileNotFoundError:
        print("Scaler files not found. Fitting new scalers (not recommended for testing)")
        # Create and fit scalers (fallback for cases where scaler files are not available)
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()
        
        # Fit and transform the data
        temp_input_scaled = input_scaler.fit_transform(temp_input)
        temp_output_scaled = output_scaler.fit_transform(temp_output)
    
    in_tensors = torch.from_numpy(temp_input_scaled).float()
    out_tensors = torch.from_numpy(temp_output_scaled).float()

    return torch.utils.data.TensorDataset(in_tensors, out_tensors), input_scaler, output_scaler

def main():
    # Set random seeds for reproducibility
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Network architecture (same as training)
    SHARED_LAYER_SIZES = [107, 114, 81]  # Optimized shared layers (3 layers)
    TEMP_LAYER_SIZES = [120, 121, 1]  # Optimized temp branch (3 layers)
    EFF_LAYER_SIZES = [38, 1]  # Optimized efficiency branch (2 layers)

    # Select GPU as default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and transform dataset
    dataset, input_scaler, output_scaler = get_dataset()

    # Create data loader for full dataset
    BATCH_SIZE = 64
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network with same architecture
    net = MultiOutputNet(
        input_size=3, 
        shared_layer_sizes=SHARED_LAYER_SIZES,
        temp_layer_sizes=TEMP_LAYER_SIZES,
        eff_layer_sizes=EFF_LAYER_SIZES
    ).to(device).float()

    # Load trained model
    try:
        # Try to load from pkl file first
        with open('Model_MultiOutput_charging.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load model state dict with proper device mapping
        if 'model_state_dict' in model_data:
            net.load_state_dict(model_data['model_state_dict'], map_location=device)
        else:
            net.load_state_dict(model_data, map_location=device)
        print("Model loaded successfully from pkl file!")
        
    except FileNotFoundError:
        # Try to load from state dict file
        try:
            net.load_state_dict(torch.load("Model_MultiOutput_charging.sd", map_location=device))
            print("Model loaded successfully from state dict file!")
        except FileNotFoundError:
            print("Error: Model files not found!")
            print("Please ensure 'Model_MultiOutput_charging.pkl' or 'Model_MultiOutput_charging.sd' exists.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            # Try loading with CPU mapping
            net.load_state_dict(torch.load("Model_MultiOutput_charging.sd", map_location='cpu'))
            print("Model loaded successfully with CPU mapping!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))
    print("Network architecture:")
    print(f"  Shared layers: {SHARED_LAYER_SIZES}")
    print(f"  Temp branch: {TEMP_LAYER_SIZES}")
    print(f"  Efficiency branch: {EFF_LAYER_SIZES}")

    # Evaluation
    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            battery_temp_pred, efficiency_pred = net(inputs)
            outputs = torch.cat([battery_temp_pred, efficiency_pred], dim=1)
            
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))

    # Concatenate all batches for full dataset predictions
    inputs_array = torch.cat(inputs_list, dim=0)
    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    
    print(f"Test shapes - y_meas: {y_meas.shape}, y_pred: {y_pred.shape}")
    
    # Calculate overall metrics
    mse_value = F.mse_loss(y_meas, y_pred).item()
    rmse_value = np.sqrt(mse_value)
    mae_value = F.l1_loss(y_meas, y_pred).item()
    print(f"Test MSE: {mse_value:.10f}")
    print(f"Test RMSE: {rmse_value:.10f}")
    print(f"Test MAE: {mae_value:.10f}")

    # Convert tensors to numpy arrays for further processing
    inputs_array = inputs_array.cpu().numpy()
    y_meas = y_meas.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    # Use StandardScaler to reverse normalize the data
    print("Using StandardScaler for reverse normalization...")
    print(f"Input scaler mean: {input_scaler.mean_}")
    print(f"Input scaler scale: {input_scaler.scale_}")
    print(f"Output scaler mean: {output_scaler.mean_}")
    print(f"Output scaler scale: {output_scaler.scale_}")

    # Reverse normalize inputs using StandardScaler
    inputs_array = input_scaler.inverse_transform(inputs_array)
    
    # Reverse normalize outputs using StandardScaler
    y_meas = output_scaler.inverse_transform(y_meas)
    y_pred = output_scaler.inverse_transform(y_pred)

    # Create DataFrame to save results
    df = pd.DataFrame(inputs_array, columns=['SOC', 'ChargingPower_kW', 'AmbientTemp'])
    output_data = {
        'y_meas_battery_temp': y_meas[:, 0], 'y_pred_battery_temp': y_pred[:, 0],
        'y_meas_efficiency': y_meas[:, 1], 'y_pred_efficiency': y_pred[:, 1]
    }
    df = pd.concat([df, pd.DataFrame(output_data)], axis=1)

    epsilon = 1e-8
    metrics = {'MAE': [], 'MSE': [], 'RMSE': [], 'MAPE': [], 'R2': []}

    # Calculate errors for each output
    output_names = ['battery_temp', 'efficiency']
    for i in range(2):
        mae = np.abs(y_meas[:, i] - y_pred[:, i])
        mse = (y_meas[:, i] - y_pred[:, i]) ** 2
        rmse = np.sqrt(mse)
        mape = np.abs((y_meas[:, i] - y_pred[:, i]) / (y_meas[:, i] + epsilon)) * 100
        r2 = r2_score(y_meas[:, i], y_pred[:, i])

        # Store each metric in the DataFrame
        df[f'MAE_{output_names[i]}'] = mae
        df[f'MSE_{output_names[i]}'] = mse
        df[f'RMSE_{output_names[i]}'] = rmse
        df[f'MAPE_{output_names[i]}'] = mape
        df[f'R2_{output_names[i]}'] = r2
        
        # Collect mean values
        metrics['MAE'].append(np.mean(mae))
        metrics['MSE'].append(np.mean(mse))
        metrics['RMSE'].append(np.mean(rmse))
        metrics['MAPE'].append(np.mean(mape))
        metrics['R2'].append(r2)

    # Print mean of each metric for all outputs
    for metric, values in metrics.items():
        print(f"Mean {metric}: {np.mean(values):.10f}")

    # Save the DataFrame to a CSV file with '04' prefix
    csv_file_path = '04full_data_y_meas_y_pred_multi_output.csv'
    df.to_csv(csv_file_path, index=False)
    print(f"Saved full data with measurements and predictions to '{csv_file_path}'")

    # Generate scatter plots for each output variable
    output_columns = ['y_meas_battery_temp', 'y_pred_battery_temp', 'y_meas_efficiency', 'y_pred_efficiency']
    colors = ['red', 'blue']
    titles = ['Battery Temperature', 'Charging Efficiency']

    for i in range(2):
        plt.figure(figsize=(8, 6))
        plt.scatter(df[f'y_meas_{output_names[i]}'], df[f'y_pred_{output_names[i]}'], 
                   color=colors[i], alpha=0.5)
        plt.title(f'Actual vs. Predicted Values for {titles[i]}\nR^2 Score: {df[f"R2_{output_names[i]}"].iloc[0]:.4f}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.axis('equal')
        plt.plot([df[f'y_meas_{output_names[i]}'].min(), df[f'y_meas_{output_names[i]}'].max()], 
                [df[f'y_meas_{output_names[i]}'].min(), df[f'y_meas_{output_names[i]}'].max()], 'k--')
        plt.savefig(f'04multi_output_scatter_plot_{output_names[i]}.png')
        plt.close()

    print("Testing completed successfully!")

if __name__ == "__main__":
    main() 