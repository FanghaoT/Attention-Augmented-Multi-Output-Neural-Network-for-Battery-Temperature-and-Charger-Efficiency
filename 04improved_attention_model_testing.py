"""
Improved Attention Multi-Output Neural Network Testing Program

This script loads the trained improved attention model and tests it on the full dataset.
The results are saved with the same format as the training program but with '04' prefix.
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os
import pickle
from sklearn.preprocessing import StandardScaler

class SelfAttention(nn.Module):
    """
    自注意力机制实现
    """
    def __init__(self, input_size, attention_size=64):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.attention_size = attention_size
        
        # Q, K, V 变换矩阵
        self.query = nn.Linear(input_size, attention_size)
        self.key = nn.Linear(input_size, attention_size)
        self.value = nn.Linear(input_size, attention_size)
        
        # 输出投影
        self.output_projection = nn.Linear(attention_size, input_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 计算 Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_size)
        attention_weights = self.softmax(attention_scores)
        
        # 应用注意力权重到 V
        attended_values = torch.matmul(attention_weights, V)
        
        # 输出投影
        output = self.output_projection(attended_values)
        attention_weights_output = F.softmax(output, dim=1)
        
        return attention_weights_output

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制实现
    """
    def __init__(self, input_size, num_heads=4, head_dim=16):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_size = input_size
        
        # 为每个头创建线性变换
        self.query_transforms = nn.ModuleList([
            nn.Linear(input_size, head_dim) for _ in range(num_heads)
        ])
        self.key_transforms = nn.ModuleList([
            nn.Linear(input_size, head_dim) for _ in range(num_heads)
        ])
        self.value_transforms = nn.ModuleList([
            nn.Linear(input_size, head_dim) for _ in range(num_heads)
        ])
        
        # 输出变换
        self.output_transform = nn.Linear(num_heads * head_dim, input_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size(0)
        head_outputs = []
        
        # 对每个头计算注意力
        for i in range(self.num_heads):
            queries = self.query_transforms[i](x)
            keys = self.key_transforms[i](x)
            values = self.value_transforms[i](x)
            
            # 计算注意力分数
            attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention_weights = self.softmax(attention_scores)
            
            # 应用注意力权重
            attended_values = torch.matmul(attention_weights, values)
            head_outputs.append(attended_values)
        
        # 连接所有头的输出
        concatenated = torch.cat(head_outputs, dim=-1)
        
        # 输出变换
        output = self.output_transform(concatenated)
        attention_weights_output = F.softmax(output, dim=1)
        
        return attention_weights_output

class PyTorchMultiheadAttention(nn.Module):
    """
    使用PyTorch内置的多头注意力机制
    """
    def __init__(self, input_size, num_heads=4):
        super(PyTorchMultiheadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        
        # 投影输入以适应注意力机制
        self.input_projection = nn.Linear(input_size, input_size)
        self.attention = nn.MultiheadAttention(input_size, num_heads, batch_first=True)
        self.output_projection = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # 投影输入
        projected_input = self.input_projection(x)
        
        # 应用多头注意力
        attended_output, attention_weights = self.attention(projected_input, projected_input, projected_input)
        
        # 输出投影
        output = self.output_projection(attended_output)
        attention_weights_output = F.softmax(output, dim=1)
        
        return attention_weights_output

class GatedAttention(nn.Module):
    """
    门控注意力机制实现
    """
    def __init__(self, input_size, attention_size=64):
        super(GatedAttention, self).__init__()
        self.input_size = input_size
        
        # 注意力计算
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, input_size)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Sigmoid(),
            nn.Linear(attention_size, input_size)
        )
        
        # 输出层
        self.output_layer = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 计算注意力
        attention_scores = self.attention(x)
        
        # 计算门控值
        gate_values = self.gate(x)
        
        # 应用门控到注意力
        gated_attention = attention_scores * gate_values
        
        # 输出注意力权重
        attention_weights = self.output_layer(gated_attention)
        
        return attention_weights

class HierarchicalAttention(nn.Module):
    """
    层次注意力机制实现
    """
    def __init__(self, input_size, feature_attention_size=32, temporal_attention_size=32):
        super(HierarchicalAttention, self).__init__()
        self.input_size = input_size
        
        # 特征级注意力
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, feature_attention_size),
            nn.Tanh(),
            nn.Linear(feature_attention_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # 时间级注意力（如果处理序列数据）
        self.temporal_attention = nn.Sequential(
            nn.Linear(input_size, temporal_attention_size),
            nn.Tanh(),
            nn.Linear(temporal_attention_size, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 特征级注意力
        feature_weights = self.feature_attention(x)
        feature_attended = x * feature_weights
        
        # 对于单时间步数据，直接返回特征注意力结果
        return feature_weights

class ImprovedAttentionMultiOutputNet(nn.Module):
    """
    改进的注意力机制多输出网络
    根据领域知识，为不同输出使用不同的注意力机制：
    - 电池温度预测：更关注环境温度
    - 效率预测：更关注充电功率
    """
    
    def __init__(self, input_size=3, attention_hidden_size=64, shared_layer_sizes=None, temp_layer_sizes=None, eff_layer_sizes=None):
        super(ImprovedAttentionMultiOutputNet, self).__init__()
        
        # 为电池温度预测设计的注意力层（更关注环境温度）
        self.temp_attention = nn.Sequential(
            nn.Linear(input_size, attention_hidden_size),
            nn.Tanh(),
            nn.Linear(attention_hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # 为效率预测设计的注意力层（更关注充电功率）
        self.eff_attention = nn.Sequential(
            nn.Linear(input_size, attention_hidden_size),
            nn.Tanh(),
            nn.Linear(attention_hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # 多头注意力机制版本（可选）
        self.temp_multihead_attention = MultiHeadAttention(input_size, num_heads=4, head_dim=8)
        self.eff_multihead_attention = MultiHeadAttention(input_size, num_heads=4, head_dim=8)
        
        # 自注意力机制版本（可选）
        self.temp_self_attention = SelfAttention(input_size, attention_size=attention_hidden_size)
        self.eff_self_attention = SelfAttention(input_size, attention_size=attention_hidden_size)
        
        # 层次注意力机制版本（可选）
        self.temp_hierarchical_attention = HierarchicalAttention(input_size, feature_attention_size=attention_hidden_size)
        self.eff_hierarchical_attention = HierarchicalAttention(input_size, feature_attention_size=attention_hidden_size)
        
        # 门控注意力机制版本（可选）
        self.temp_gated_attention = GatedAttention(input_size, attention_size=attention_hidden_size)
        self.eff_gated_attention = GatedAttention(input_size, attention_size=attention_hidden_size)
        
        # 电池温度预测的共享特征提取层
        temp_shared_layers = []
        temp_in_features = input_size
        for i, out_features in enumerate(shared_layer_sizes):
            temp_shared_layers.append(nn.Linear(temp_in_features, out_features))
            temp_shared_layers.append(nn.ReLU())
            temp_in_features = out_features
        
        self.temp_shared_layers = nn.Sequential(*temp_shared_layers)
        
        # 效率预测的共享特征提取层
        eff_shared_layers = []
        eff_in_features = input_size
        for i, out_features in enumerate(shared_layer_sizes):
            eff_shared_layers.append(nn.Linear(eff_in_features, out_features))
            eff_shared_layers.append(nn.ReLU())
            eff_in_features = out_features
        
        self.eff_shared_layers = nn.Sequential(*eff_shared_layers)
        
        # 电池温度预测分支
        temp_layers = []
        temp_in_features = shared_layer_sizes[-1]  # 来自共享层的输出
        for i, out_features in enumerate(temp_layer_sizes):
            temp_layers.append(nn.Linear(temp_in_features, out_features))
            if i < len(temp_layer_sizes) - 1:  # 不是最后一层
                temp_layers.append(nn.ReLU())
            temp_in_features = out_features
        
        self.temp_branch = nn.Sequential(*temp_layers)
        
        # 效率预测分支 (包含电池温度作为输入)
        eff_layers = []
        eff_in_features = shared_layer_sizes[-1] + 1  # 共享特征 + 电池温度
        for i, out_features in enumerate(eff_layer_sizes):
            eff_layers.append(nn.Linear(eff_in_features, out_features))
            if i < len(eff_layer_sizes) - 1:  # 不是最后一层
                eff_layers.append(nn.ReLU())
            eff_in_features = out_features
        
        self.efficiency_branch = nn.Sequential(*eff_layers)
    
    def forward(self, x, attention_type="standard"):
        """
        前向传播函数
        Args:
            x: 输入张量
            attention_type: 注意力机制类型 ("standard", "multihead", "self", "hierarchical", "gated")
        """
        if attention_type == "multihead":
            # 使用多头注意力机制
            temp_attention_weights = self.temp_multihead_attention(x)
            temp_attended_input = x * temp_attention_weights
            
            eff_attention_weights = self.eff_multihead_attention(x)
            eff_attended_input = x * eff_attention_weights
        elif attention_type == "self":
            # 使用自注意力机制
            temp_attention_weights = self.temp_self_attention(x)
            temp_attended_input = x * temp_attention_weights
            
            eff_attention_weights = self.eff_self_attention(x)
            eff_attended_input = x * eff_attention_weights
        elif attention_type == "hierarchical":
            # 使用层次注意力机制
            temp_attention_weights = self.temp_hierarchical_attention(x)
            temp_attended_input = x * temp_attention_weights
            
            eff_attention_weights = self.eff_hierarchical_attention(x)
            eff_attended_input = x * eff_attention_weights
        elif attention_type == "gated":
            # 使用门控注意力机制
            temp_attention_weights = self.temp_gated_attention(x)
            temp_attended_input = x * temp_attention_weights
            
            eff_attention_weights = self.eff_gated_attention(x)
            eff_attended_input = x * eff_attention_weights
        else:
            # 标准注意力机制
            temp_attention_weights = self.temp_attention(x)
            temp_attended_input = x * temp_attention_weights
            
            eff_attention_weights = self.eff_attention(x)
            eff_attended_input = x * eff_attention_weights
        
        # 使用电池温度注意力权重提取共享特征
        temp_shared_features = self.temp_shared_layers(temp_attended_input)
        
        # 使用效率注意力权重提取共享特征
        eff_shared_features = self.eff_shared_layers(eff_attended_input)
        
        # 预测电池温度
        battery_temp = self.temp_branch(temp_shared_features)
        
        # 预测效率（使用不同的共享特征）
        efficiency_input = torch.cat([eff_shared_features, battery_temp], dim=1)
        efficiency = self.efficiency_branch(efficiency_input)
        
        return battery_temp, efficiency, temp_attention_weights, eff_attention_weights

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
        with open('scaler_improved_attention_input.pkl', 'rb') as f:
            input_scaler = pickle.load(f)
        with open('scaler_improved_attention_output.pkl', 'rb') as f:
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
    random_seed = 999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Select GPU as default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and transform dataset
    dataset, input_scaler, output_scaler = get_dataset()

    # Create data loader for full dataset
    BATCH_SIZE = 64
    kwargs = {'num_workers': 0, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Load model parameters from saved model or fallback to optimized values
    model_params = None
    try:
        # Try to load from pkl file first
        with open('Model_Improved_Attention_charging.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract model parameters from saved model
        if 'model_params' in model_data:
            model_params = model_data['model_params']
            print("Loaded model parameters from saved model")
        else:
            # Fallback to optimized parameters
            model_params = {
                'attention_hidden_size': 95,
                'shared_layer_sizes': [107, 114, 81],
                'temp_layer_sizes': [120, 121, 1],
                'eff_layer_sizes': [38, 1]
            }
            print("Using fallback optimized parameters")
            
    except FileNotFoundError:
        # Fallback to optimized parameters
        model_params = {
            'attention_hidden_size': 95,
            'shared_layer_sizes': [107, 114, 81],
            'temp_layer_sizes': [120, 121, 1],
            'eff_layer_sizes': [38, 1]
        }
        print("Model file not found, using fallback optimized parameters")

    # Setup network with correct architecture
    net = ImprovedAttentionMultiOutputNet(
        input_size=3, 
        attention_hidden_size=model_params['attention_hidden_size'],
        shared_layer_sizes=model_params['shared_layer_sizes'],
        temp_layer_sizes=model_params['temp_layer_sizes'],
        eff_layer_sizes=model_params['eff_layer_sizes']
    ).to(device).float()

    # Load trained model
    try:
        # Try to load from pkl file first
        with open('Model_Improved_Attention_charging.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        # Load model state dict (without map_location for pkl files)
        if 'model_state_dict' in model_data:
            net.load_state_dict(model_data['model_state_dict'])
        else:
            net.load_state_dict(model_data)
        print("Model loaded successfully from pkl file!")
        
    except FileNotFoundError:
        # Try to load from state dict file
        try:
            net.load_state_dict(torch.load("Model_Improved_Attention_charging.sd", map_location=device))
            print("Model loaded successfully from state dict file!")
        except FileNotFoundError:
            print("Error: Model files not found!")
            print("Please ensure 'Model_Improved_Attention_charging.pkl' or 'Model_Improved_Attention_charging.sd' exists.")
            return
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        try:
            # Try loading with CPU mapping
            net.load_state_dict(torch.load("Model_Improved_Attention_charging.sd", map_location='cpu'))
            print("Model loaded successfully with CPU mapping!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))
    print("Network architecture:")
    print(f"  Attention hidden size: {model_params['attention_hidden_size']}")
    print(f"  Shared layers: {model_params['shared_layer_sizes']}")
    print(f"  Temp branch: {model_params['temp_layer_sizes']}")
    print(f"  Efficiency branch: {model_params['eff_layer_sizes']}")

    # Evaluation
    net.eval()
    inputs_list = []
    y_meas = []
    y_pred = []
    temp_attention_weights_list = []
    eff_attention_weights_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            battery_temp_pred, efficiency_pred, temp_attention_weights, eff_attention_weights = net(inputs)
            outputs = torch.cat([battery_temp_pred, efficiency_pred], dim=1)
            
            y_pred.append(outputs)
            y_meas.append(labels.to(device))
            inputs_list.append(inputs.to(device))
            temp_attention_weights_list.append(temp_attention_weights)
            eff_attention_weights_list.append(eff_attention_weights)

    # Concatenate all batches for full dataset predictions
    inputs_array = torch.cat(inputs_list, dim=0)
    y_meas = torch.cat(y_meas, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    temp_attention_weights_array = torch.cat(temp_attention_weights_list, dim=0)
    eff_attention_weights_array = torch.cat(eff_attention_weights_list, dim=0)
    
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
    temp_attention_weights_array = temp_attention_weights_array.cpu().numpy()
    eff_attention_weights_array = eff_attention_weights_array.cpu().numpy()

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
    attention_data = {
        'temp_attention_SOC': temp_attention_weights_array[:, 0],
        'temp_attention_Power': temp_attention_weights_array[:, 1],
        'temp_attention_AmbientTemp': temp_attention_weights_array[:, 2],
        'eff_attention_SOC': eff_attention_weights_array[:, 0],
        'eff_attention_Power': eff_attention_weights_array[:, 1],
        'eff_attention_AmbientTemp': eff_attention_weights_array[:, 2]
    }
    df = pd.concat([df, pd.DataFrame(output_data), pd.DataFrame(attention_data)], axis=1)

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
    csv_file_path = '04full_data_y_meas_y_pred_improved_attention.csv'
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
        plt.savefig(f'04improved_attention_scatter_plot_{output_names[i]}.png')
        plt.close()

    # Generate attention weights analysis plot
    plt.figure(figsize=(15, 10))
    
    # Mean attention weights for temperature prediction
    plt.subplot(2, 3, 1)
    mean_temp_attention = np.mean(temp_attention_weights_array, axis=0)
    plt.bar(['SOC', 'Power', 'AmbientTemp'], mean_temp_attention, color=['blue', 'green', 'red'])
    plt.title('Mean Temp Attention Weights')
    plt.ylabel('Attention Weight')
    
    # Mean attention weights for efficiency prediction
    plt.subplot(2, 3, 2)
    mean_eff_attention = np.mean(eff_attention_weights_array, axis=0)
    plt.bar(['SOC', 'Power', 'AmbientTemp'], mean_eff_attention, color=['blue', 'green', 'red'])
    plt.title('Mean Eff Attention Weights')
    plt.ylabel('Attention Weight')
    
    # Histogram of temperature attention weights
    plt.subplot(2, 3, 3)
    plt.hist(temp_attention_weights_array[:, 0], alpha=0.7, label='SOC', bins=30)
    plt.hist(temp_attention_weights_array[:, 1], alpha=0.7, label='Power', bins=30)
    plt.hist(temp_attention_weights_array[:, 2], alpha=0.7, label='AmbientTemp', bins=30)
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Histogram of Temp Attention Weights')
    plt.legend()
    
    # Histogram of efficiency attention weights
    plt.subplot(2, 3, 4)
    plt.hist(eff_attention_weights_array[:, 0], alpha=0.7, label='SOC', bins=30)
    plt.hist(eff_attention_weights_array[:, 1], alpha=0.7, label='Power', bins=30)
    plt.hist(eff_attention_weights_array[:, 2], alpha=0.7, label='AmbientTemp', bins=30)
    plt.xlabel('Attention Weight')
    plt.ylabel('Frequency')
    plt.title('Histogram of Eff Attention Weights')
    plt.legend()
    
    # Scatter plot: temperature attention weights vs prediction accuracy
    plt.subplot(2, 3, 5)
    plt.scatter(temp_attention_weights_array[:, 2], df['R2_battery_temp'], alpha=0.5, label='AmbientTemp')
    plt.xlabel('AmbientTemp Attention Weight')
    plt.ylabel('R² Score (Battery Temp)')
    plt.title('Temp Attention vs Battery Temp Accuracy')
    plt.legend()
    
    # Scatter plot: efficiency attention weights vs prediction accuracy
    plt.subplot(2, 3, 6)
    plt.scatter(eff_attention_weights_array[:, 1], df['R2_efficiency'], alpha=0.5, label='Power')
    plt.xlabel('Power Attention Weight')
    plt.ylabel('R² Score (Efficiency)')
    plt.title('Eff Attention vs Efficiency Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('04improved_attention_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Improved attention weights analysis saved to '04improved_attention_weights_analysis.png'")
    print("Testing completed successfully!")

if __name__ == "__main__":
    main() 