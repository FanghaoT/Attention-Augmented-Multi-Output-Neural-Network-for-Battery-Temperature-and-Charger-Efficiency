
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
        
        # 真正的共享特征提取层（温度和效率预测共享相同的权重）
        shared_layers = []
        shared_in_features = input_size
        for i, out_features in enumerate(shared_layer_sizes):
            shared_layers.append(nn.Linear(shared_in_features, out_features))
            shared_layers.append(nn.ReLU())
            shared_in_features = out_features
        
        self.shared_layers = nn.Sequential(*shared_layers)
        
        # 电池温度预测分支
        temp_layers = []
        temp_in_features = shared_layer_sizes[-1]  # 来自共享层的输出
        for i, out_features in enumerate(temp_layer_sizes):
            temp_layers.append(nn.Linear(temp_in_features, out_features))
            if i < len(temp_layer_sizes) - 1:  # 不是最后一层
                temp_layers.append(nn.ReLU())
            temp_in_features = out_features
        
        self.temp_branch = nn.Sequential(*temp_layers)
        
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
        temp_attention_weights = self.temp_attention(x)
        temp_attended_input = x * temp_attention_weights
        
        # 使用电池温度注意力权重提取共享特征
        temp_shared_features = self.shared_layers(temp_attended_input)
        
        # 预测电池温度
        battery_temp = self.temp_branch(temp_shared_features)
        
        # 创建用于效率预测的修改输入：用预测的电池温度替换环境温度
        # 原始输入: [SOC, ChargingPower, AmbientTemp]
        # 修改后: [SOC, ChargingPower, PredictedBatteryTemp]
        modified_input = torch.cat([x[:, :2], battery_temp], dim=1)  # 保留SOC和Power，用预测温度替换环境温度
        
        # 效率预测路径的注意力机制（基于修改后的输入）
        # 注意：这里使用modified_input是关键，因为效率attention需要基于实际的输入语义
        # modified_input = [SOC, Power, PredictedBatteryTemp]，而不是原始的[SOC, Power, AmbientTemp]
       
        eff_attention_weights = self.eff_attention(modified_input)
        
        eff_attended_input = modified_input * eff_attention_weights
        
        # 使用效率注意力权重提取共享特征（使用相同的共享层）
        eff_shared_features = self.shared_layers(eff_attended_input)
        
        # 预测效率
        efficiency = self.efficiency_branch(eff_shared_features)
        
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
    
    # Create and fit scalers
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()
    
    # Fit and transform the data
    temp_input_scaled = input_scaler.fit_transform(temp_input)
    temp_output_scaled = output_scaler.fit_transform(temp_output)
    
    in_tensors = torch.from_numpy(temp_input_scaled).float()
    out_tensors = torch.from_numpy(temp_output_scaled).float()

    return torch.utils.data.TensorDataset(in_tensors, out_tensors), input_scaler, output_scaler

def custom_loss_function(y1_pred, y1_true, y2_pred, y2_true):
    # 电池温度损失
    temp_loss = F.mse_loss(y1_pred, y1_true)
    
    # 效率损失
    efficiency_loss = F.mse_loss(y2_pred, y2_true)
    
    # 总损失 - 只使用基本的MSE损失
    total_loss = 0.5* (temp_loss + efficiency_loss)
    
    return total_loss, temp_loss, efficiency_loss

# Config the model training
def main():
    # Set random seeds for reproducibility
    random_seed = 999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Hyperparameters (based on optimized values from Optuna optimization)
    NUM_EPOCH = 600
    BATCH_SIZE = 64
    DECAY_EPOCH = 100
    DECAY_RATIO = 0.8
    LR_INI = 0.004657850533025872  # Optimized learning rate from Optuna

    
    # Optimized network architecture from Optuna
    ATTENTION_HIDDEN_SIZE = 95  # Optimized attention hidden size
    SHARED_LAYER_SIZES = [107, 114, 81]  # Optimized shared layers (3 layers)
    TEMP_LAYER_SIZES = [120, 121, 1]  # Optimized temp branch (1 layer)
    EFF_LAYER_SIZES = [38, 1]  # Optimized efficiency branch (3 layers)

    # Select GPU as default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and transform dataset
    dataset, input_scaler, output_scaler = get_dataset()

    # Split the dataset with fixed random seed
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size], 
        generator=torch.Generator().manual_seed(random_seed)
    )

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    # Setup network with improved architecture
    net = ImprovedAttentionMultiOutputNet(
        input_size=3, 
        attention_hidden_size=ATTENTION_HIDDEN_SIZE,
        shared_layer_sizes=SHARED_LAYER_SIZES,
        temp_layer_sizes=TEMP_LAYER_SIZES,
        eff_layer_sizes=EFF_LAYER_SIZES
    ).to(device).float()

    # Log the number of parameters
    print("Number of parameters: ", count_parameters(net))

    # Setup optimizer with optimized parameters (Adam)
    optimizer = optim.Adam(net.parameters(), lr=LR_INI) 

    # Define lists to store losses
    train_loss_list = np.zeros([NUM_EPOCH, 3])  # [total_loss, temp_loss, efficiency_loss]
    valid_loss_list = np.zeros([NUM_EPOCH, 3])
    

    # Train the network
    for epoch_i in range(NUM_EPOCH):
        # Train for one epoch
        epoch_train_total_loss = 0
        epoch_train_temp_loss = 0
        epoch_train_efficiency_loss = 0

        net.train()
        optimizer.param_groups[0]['lr'] = LR_INI * (DECAY_RATIO ** (epoch_i // DECAY_EPOCH))
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            
            # Forward pass
            battery_temp_pred, efficiency_pred, temp_attention_weights, eff_attention_weights = net(inputs)
            
            # Split labels
            battery_temp_true = labels[:, 0:1]
            efficiency_true = labels[:, 1:2]
            
            # Calculate loss
            total_loss, temp_loss, efficiency_loss = custom_loss_function(
                battery_temp_pred, battery_temp_true, 
                efficiency_pred, efficiency_true
            )
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_train_total_loss += total_loss.item()
            epoch_train_temp_loss += temp_loss.item()
            epoch_train_efficiency_loss += efficiency_loss.item()
        
        # Validation
        net.eval()
        epoch_valid_total_loss = 0
        epoch_valid_temp_loss = 0
        epoch_valid_efficiency_loss = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).float()
                battery_temp_pred, efficiency_pred, _, _ = net(inputs)
                battery_temp_true = labels[:, 0:1]
                efficiency_true = labels[:, 1:2]
                total_loss, temp_loss, efficiency_loss = custom_loss_function(
                    battery_temp_pred, battery_temp_true,
                    efficiency_pred, efficiency_true
                )
                epoch_valid_total_loss += total_loss.item()
                epoch_valid_temp_loss += temp_loss.item()
                epoch_valid_efficiency_loss += efficiency_loss.item()
        
        # Calculate average losses
        train_count = len(train_loader)
        valid_count = len(valid_loader)
        train_loss_list[epoch_i, 0] = epoch_train_total_loss / train_count
        train_loss_list[epoch_i, 1] = epoch_train_temp_loss / train_count
        train_loss_list[epoch_i, 2] = epoch_train_efficiency_loss / train_count
        valid_loss_list[epoch_i, 0] = epoch_valid_total_loss / valid_count
        valid_loss_list[epoch_i, 1] = epoch_valid_temp_loss / valid_count
        valid_loss_list[epoch_i, 2] = epoch_valid_efficiency_loss / valid_count
        

        
        # Print progress
        if (epoch_i + 1) % 100 == 0:
            print(f"Epoch [{epoch_i+1}/{NUM_EPOCH}], Train Loss: {train_loss_list[epoch_i, 0]:.6f}, Valid Loss: {valid_loss_list[epoch_i, 0]:.6f}")
            print(f"  Temp Train Loss: {train_loss_list[epoch_i, 1]:.6f}, Temp Valid Loss: {valid_loss_list[epoch_i, 1]:.6f}")
            print(f"  Eff Train Loss: {train_loss_list[epoch_i, 2]:.6f}, Eff Valid Loss: {valid_loss_list[epoch_i, 2]:.6f}")

    # Save the model parameters
    torch.save(net.state_dict(), "Model_Improved_Attention_charging.sd")
    print("Training finished! Model is saved!")
    

    
    # Save model and scalers as pkl files
    model_data = {
        'model_state_dict': net.state_dict(),
        'model_class': ImprovedAttentionMultiOutputNet,
        'model_params': {
            'input_size': 3, 
            'attention_hidden_size': ATTENTION_HIDDEN_SIZE,
            'shared_layer_sizes': SHARED_LAYER_SIZES,
            'temp_layer_sizes': TEMP_LAYER_SIZES,
            'eff_layer_sizes': EFF_LAYER_SIZES
        }
    }
    
    with open('Model_Improved_Attention_charging.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    with open('scaler_improved_attention_input.pkl', 'wb') as f:
        pickle.dump(input_scaler, f)
    
    with open('scaler_improved_attention_output.pkl', 'wb') as f:
        pickle.dump(output_scaler, f)
    
    print("Model and scalers saved as pkl files!")
    
    np.savetxt('train_loss_improved_attention.csv', train_loss_list, delimiter=',')
    np.savetxt('valid_loss_improved_attention.csv', valid_loss_list, delimiter=',')

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
    print(f"Temp attention weights shape: {temp_attention_weights_array.shape}")
    print(f"Eff attention weights shape: {eff_attention_weights_array.shape}")
    
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
    
    # Add attention weights
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
    
    # 添加额外的分析信息
    print("\nDetailed Analysis:")
    print(f"  Battery Temp R² std: {np.std([r2_score(y_meas[i::len(y_meas)//2, 0], y_pred[i::len(y_pred)//2, 0]) for i in range(len(y_meas)//2)]):.6f}")
    print(f"  Efficiency R² std: {np.std([r2_score(y_meas[i::len(y_meas)//2, 1], y_pred[i::len(y_pred)//2, 1]) for i in range(len(y_meas)//2)]):.6f}")
    print(f"  Battery Temp variance - True: {np.var(y_meas[:, 0]):.6f}, Pred: {np.var(y_pred[:, 0]):.6f}")
    print(f"  Efficiency variance - True: {np.var(y_meas[:, 1]):.6f}, Pred: {np.var(y_pred[:, 1]):.6f}")

    # Save the DataFrame to a CSV file
    csv_file_path = 'full_data_y_meas_y_pred_improved_attention.csv'
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
        plt.savefig(f'improved_attention_scatter_plot_{output_names[i]}.png')
        plt.close()

    # Analyze attention weights
    print("\nAttention Weights Analysis:")
    print("Battery Temperature Prediction:")
    print(f"  Mean attention weight for SOC: {np.mean(temp_attention_weights_array[:, 0]):.4f}")
    print(f"  Mean attention weight for Power: {np.mean(temp_attention_weights_array[:, 1]):.4f}")
    print(f"  Mean attention weight for AmbientTemp: {np.mean(temp_attention_weights_array[:, 2]):.4f}")
    
    print("Efficiency Prediction:")
    print(f"  Mean attention weight for SOC: {np.mean(eff_attention_weights_array[:, 0]):.4f}")
    print(f"  Mean attention weight for Power: {np.mean(eff_attention_weights_array[:, 1]):.4f}")
    print(f"  Mean attention weight for AmbientTemp: {np.mean(eff_attention_weights_array[:, 2]):.4f}")
    
    # Create attention weights visualization
    plt.figure(figsize=(15, 10))
    
    # Box plot of temperature attention weights
    plt.subplot(2, 3, 1)
    plt.boxplot([temp_attention_weights_array[:, 0], temp_attention_weights_array[:, 1], temp_attention_weights_array[:, 2]], 
                labels=['SOC', 'Power', 'AmbientTemp'])
    plt.title('Distribution of Temp Attention Weights')
    plt.ylabel('Attention Weight')
    
    # Box plot of efficiency attention weights
    plt.subplot(2, 3, 2)
    plt.boxplot([eff_attention_weights_array[:, 0], eff_attention_weights_array[:, 1], eff_attention_weights_array[:, 2]], 
                labels=['SOC', 'Power', 'AmbientTemp'])
    plt.title('Distribution of Eff Attention Weights')
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
    plt.savefig('improved_attention_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Improved attention weights analysis saved to 'improved_attention_weights_analysis.png'")

    # --- Save validation set predictions/results to CSV ---
    val_inputs_list = []
    val_y_meas = []
    val_y_pred = []
    val_temp_attention_weights_list = []
    val_eff_attention_weights_list = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()
            battery_temp_pred, efficiency_pred, temp_attention_weights, eff_attention_weights = net(inputs)
            outputs = torch.cat([battery_temp_pred, efficiency_pred], dim=1)
            
            val_y_pred.append(outputs)
            val_y_meas.append(labels.to(device))
            val_inputs_list.append(inputs.to(device))
            val_temp_attention_weights_list.append(temp_attention_weights)
            val_eff_attention_weights_list.append(eff_attention_weights)

    val_inputs_array = torch.cat(val_inputs_list, dim=0)
    val_y_meas = torch.cat(val_y_meas, dim=0)
    val_y_pred = torch.cat(val_y_pred, dim=0)
    val_temp_attention_weights_array = torch.cat(val_temp_attention_weights_list, dim=0)
    val_eff_attention_weights_array = torch.cat(val_eff_attention_weights_list, dim=0)

    # Reverse normalization
    val_inputs_array = val_inputs_array.cpu().numpy()
    val_y_meas = val_y_meas.cpu().numpy()
    val_y_pred = val_y_pred.cpu().numpy()
    val_temp_attention_weights_array = val_temp_attention_weights_array.cpu().numpy()
    val_eff_attention_weights_array = val_eff_attention_weights_array.cpu().numpy()

    val_inputs_array = input_scaler.inverse_transform(val_inputs_array)
    val_y_meas = output_scaler.inverse_transform(val_y_meas)
    val_y_pred = output_scaler.inverse_transform(val_y_pred)

    val_df = pd.DataFrame(val_inputs_array, columns=['SOC', 'ChargingPower_kW', 'AmbientTemp'])
    val_output_data = {
        'y_meas_battery_temp': val_y_meas[:, 0], 'y_pred_battery_temp': val_y_pred[:, 0],
        'y_meas_efficiency': val_y_meas[:, 1], 'y_pred_efficiency': val_y_pred[:, 1]
    }
    val_attention_data = {
        'temp_attention_SOC': val_temp_attention_weights_array[:, 0],
        'temp_attention_Power': val_temp_attention_weights_array[:, 1],
        'temp_attention_AmbientTemp': val_temp_attention_weights_array[:, 2],
        'eff_attention_SOC': val_eff_attention_weights_array[:, 0],
        'eff_attention_Power': val_eff_attention_weights_array[:, 1],
        'eff_attention_AmbientTemp': val_eff_attention_weights_array[:, 2]
    }
    val_df = pd.concat([val_df, pd.DataFrame(val_output_data), pd.DataFrame(val_attention_data)], axis=1)
    val_csv_file_path = 'validation_data_y_meas_y_pred_improved_attention.csv'
    val_df.to_csv(val_csv_file_path, index=False)
    print(f"Saved validation data with measurements and predictions to '{val_csv_file_path}'")

if __name__ == "__main__":
    main()