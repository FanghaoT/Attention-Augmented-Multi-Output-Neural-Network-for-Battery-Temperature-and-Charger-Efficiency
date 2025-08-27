"""
Optuna example that optimizes multi-output neural networks using PyTorch.

This script optimizes the hyperparameters for the multi-output neural network
that predicts both battery temperature and charging efficiency.
"""

import os
import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 1000

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

def get_dataset():
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

    return torch.utils.data.TensorDataset(in_tensors, out_tensors)

def get_dataloader():
    dataset = get_dataset()
    train_size = int(0.8 * len(dataset))
    valid_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(999)
    )
    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    return train_loader, valid_loader, train_size, valid_size

def custom_loss_function(y1_pred, y1_true, y2_pred, y2_true, alpha=0.5):
    """
    自定义损失函数
    alpha: 控制两个输出损失之间的权重
    """
    # 电池温度损失
    temp_loss = F.mse_loss(y1_pred, y1_true)
    
    # 效率损失
    efficiency_loss = F.mse_loss(y2_pred, y2_true)
    
    # 总损失
    total_loss = alpha * temp_loss + (1 - alpha) * efficiency_loss
    
    return total_loss, temp_loss, efficiency_loss

def define_model(trial):
    # 共享层参数 - 扩大范围以匹配实际使用
    n_shared_layers = trial.suggest_int("n_shared_layers", 1, 3)  # 允耸3层
    shared_layer_sizes = []
    in_features = 3  # 输入特征数
    for i in range(n_shared_layers):
        out_features = trial.suggest_int(f"shared_units_l{i}", 16, 128)  # 扩大范围
        shared_layer_sizes.append(out_features)
        in_features = out_features
    
    # 电池温度分支参数
    n_temp_layers = trial.suggest_int("n_temp_layers", 1, 3)  # 允耸3层
    temp_layer_sizes = []
    for i in range(n_temp_layers):
        if i == n_temp_layers - 1:
            out_features = 1  # 最终输出
        else:
            out_features = trial.suggest_int(f"temp_units_l{i}", 8, 128)  # 扩大范围
        temp_layer_sizes.append(out_features)
    
    # 效率分支参数
    n_eff_layers = trial.suggest_int("n_eff_layers", 1, 3)  # 允耸3层
    eff_layer_sizes = []
    for i in range(n_eff_layers):
        if i == n_eff_layers - 1:
            out_features = 1  # 最终输出
        else:
            out_features = trial.suggest_int(f"eff_units_l{i}", 8, 128)  # 扩大范围
        eff_layer_sizes.append(out_features)
    
    return MultiOutputNet(
        input_size=3,
        shared_layer_sizes=shared_layer_sizes,
        temp_layer_sizes=temp_layer_sizes,
        eff_layer_sizes=eff_layer_sizes
    )

def objective(trial):
    try:
        # 首先验证模型架构
        model = define_model(trial).to(DEVICE)
        
        # 测试模型前向传播
        test_input = torch.randn(10, 3).to(DEVICE)
        try:
            with torch.no_grad():
                battery_temp_pred, efficiency_pred = model(test_input)
                if torch.isnan(battery_temp_pred).any() or torch.isnan(efficiency_pred).any():
                    raise ValueError("Model produces NaN outputs")
        except Exception as e:
            print(f"Model architecture test failed: {e}")
            return float('inf')
        
        # 优化器参数 - 扩大学习率范围
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # 扩大范围
        
        # 损失函数权重
        alpha = trial.suggest_float("alpha", 0.3, 0.9)  # 扩大范围
        
        # 早停参数
        patience = 100
        
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        train_loader, valid_loader, train_size, valid_size = get_dataloader()

        # 添加梯度裁剪
        max_grad_norm = 1.0
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        # 添加训练历史记录
        train_losses = []
        valid_losses = []

        for epoch in range(EPOCHS):
            model.train()
            # 更温和的学习率衰减
            current_lr = lr * (0.8 ** (epoch // 100))
            optimizer.param_groups[0]['lr'] = current_lr
            
            epoch_train_loss = 0
            batch_count = 0
            
            for data, target in train_loader:
                try:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    battery_temp_pred, efficiency_pred = model(data)
                    
                    # Split labels
                    battery_temp_true = target[:, 0:1]
                    efficiency_true = target[:, 1:2]
                    
                    # Calculate loss
                    loss, _, _ = custom_loss_function(
                        battery_temp_pred, battery_temp_true, 
                        efficiency_pred, efficiency_true, 
                        alpha=alpha
                    )
                    
                    # 检查损失是否为NaN或无穷大
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(f"Training loss is {loss.item()}")
                    
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    epoch_train_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    raise e
            
            avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            valid_loss = 0
            valid_batch_count = 0
            
            with torch.no_grad():
                for data, target in valid_loader:
                    try:
                        data, target = data.to(DEVICE), target.to(DEVICE)
                        battery_temp_pred, efficiency_pred = model(data)
                        
                        battery_temp_true = target[:, 0:1]
                        efficiency_true = target[:, 1:2]
                        
                        loss, _, _ = custom_loss_function(
                            battery_temp_pred, battery_temp_true, 
                            efficiency_pred, efficiency_true, 
                            alpha=alpha
                        )
                        valid_loss += loss.item()
                        valid_batch_count += 1
                        
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        raise e
            
            avg_valid_loss = valid_loss / valid_batch_count if valid_batch_count > 0 else float('inf')
            valid_losses.append(avg_valid_loss)
            
            # 检查验证损失是否为NaN或无穷大
            if np.isnan(avg_valid_loss) or np.isinf(avg_valid_loss):
                raise ValueError(f"Validation loss is {avg_valid_loss}")
            
            # 早停机制
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best loss: {best_valid_loss:.6f}")
                break
            
            # 每50个epoch打印一次进度
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.6f}, Valid Loss: {avg_valid_loss:.6f}")
            
            trial.report(avg_valid_loss, epoch)
            
            # 更保守的剪枝策略 - 只在损失没有改善时才考虑剪枝
            if trial.should_prune():
                print(f"Trial pruned at epoch {epoch} with loss: {avg_valid_loss:.6f}")
                raise optuna.exceptions.TrialPruned()
        
        print(f"Trial completed with best validation loss: {best_valid_loss:.6f}")
        return best_valid_loss
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        print(f"Trial parameters: {trial.params}")
        # 返回一个很大的损失值而不是抛出异常，这样Optuna可以继续
        return float('inf')
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"GPU out of memory for trial with parameters: {trial.params}")
            return float('inf')
        else:
            print(f"Runtime error in trial: {e}")
            print(f"Trial parameters: {trial.params}")
            return float('inf')

def test_model_architecture():
    """测试网络架构是否正确"""
    print("Testing model architecture...")
    try:
        # 测试不同的参数组合
        test_configs = [
            {
                'shared_layer_sizes': [64],
                'temp_layer_sizes': [1],
                'eff_layer_sizes': [1]
            },
            {
                'shared_layer_sizes': [128, 64],
                'temp_layer_sizes': [32, 1],
                'eff_layer_sizes': [32, 1]
            },
        ]
        
        for i, config in enumerate(test_configs):
            print(f"Testing configuration {i+1}: {config}")
            model = MultiOutputNet(
                input_size=3,
                shared_layer_sizes=config['shared_layer_sizes'],
                temp_layer_sizes=config['temp_layer_sizes'],
                eff_layer_sizes=config['eff_layer_sizes']
            )
            
            # 创建测试数据
            test_input = torch.randn(10, 3)  # batch_size=10, input_size=3
            battery_temp_pred, efficiency_pred = model(test_input)
            
            print(f"  Input shape: {test_input.shape}")
            print(f"  Battery temp output shape: {battery_temp_pred.shape}")
            print(f"  Efficiency output shape: {efficiency_pred.shape}")
            print("  ✓ Configuration works!")
            
    except Exception as e:
        print(f"  ✗ Configuration failed: {e}")
        raise e

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 首先测试网络架构
    test_model_architecture()
    print("\nStarting Optuna optimization...")
    
    # 选择是否启用剪枝
    ENABLE_PRUNING = True  # 设置为False来禁用剪枝
    
    if ENABLE_PRUNING:
        # 使用更保守的剪枝策略
        from optuna.pruners import MedianPruner
        study = optuna.create_study(
            direction="minimize",
            pruner=MedianPruner(
                n_startup_trials=10,  # 增加启动试验数
                n_warmup_steps=50,    # 增加预热步数
                interval_steps=10      # 减少报告间隔
            )
        )
    else:
        # 禁用剪枝
        study = optuna.create_study(direction="minimize")
        print("Pruning disabled - all trials will run to completion")
    
    study.optimize(objective, n_trials=1000, timeout=3600*20)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("  Number of failed trials: ", len(failed_trials))
    
    if len(complete_trials) > 0:
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best parameters
        best_params = trial.params
        np.save('best_params_multi_output.npy', best_params)
        print("Best parameters saved to 'best_params_multi_output.npy'")
        
        # 打印最佳参数摘要
        print("\nBest Parameters Summary:")
        print(f"  Learning Rate: {trial.params.get('lr', 'N/A')}")
        print(f"  Loss Weight (alpha): {trial.params.get('alpha', 'N/A')}")
        print(f"  Best Validation Loss: {trial.value:.6f}")
        
    else:
        print("No completed trials found. Check the error messages above.") 