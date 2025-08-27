# Attention-Augmented Multi-Output Neural Network for Battery Temperature and Charger Efficiency

## Overview

This repository contains an advanced machine learning system that utilizes attention-augmented neural networks to simultaneously predict battery temperature and charger efficiency.

## Features

- Multi-Output Neural Network for simultaneous prediction
- Attention mechanism for better feature selection
- Hyperparameter optimization using Optuna
- Comprehensive data preprocessing pipeline
- Extensive testing framework

## Project Structure

```
├── 01data_visual_preproces.py          # Data preprocessing
├── 02improved_attention_model.py       # Attention model
├── 02multi_output_nn_training.py       # Multi-output training
├── 03Optuna_improved_attention.py      # Attention model optimization
├── 03Optuna_multi_output.py            # Multi-output optimization
├── 04improved_attention_model_testing.py # Model testing
├── 04multi_output_nn_testing.py        # Testing framework
├── InputData/                          # Input datasets
├── OutputData/                         # Processed data
└── OutputANN/                          # Model outputs
```

## Installation

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn optuna tensorflow keras
```

## Usage

1. **Data Preprocessing**: `python 01data_visual_preproces.py`
2. **Model Training**: `python 02improved_attention_model.py`
3. **Optimization**: `python 03Optuna_improved_attention.py`
4. **Testing**: `python 04improved_attention_model_testing.py`

## License

MIT License
