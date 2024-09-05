import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import yaml





def load_config(config_file):
    """
    Loads the configuration from a YAML file.

    Parameters:
    config_file (str): Path to the configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def is_continuous(timestamps: pd.Index) -> bool:
    """
    Check if the timestamps in the window are continuous.

    Parameters:
    timestamps (pd.Index): Index of timestamps.

    Returns:
    bool: True if timestamps are continuous, False otherwise.
    """
    return all((timestamps[i + 1] - timestamps[i]) == 1 for i in range(len(timestamps) - 1))

def create_windows(df: pd.DataFrame, window_size: int, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create windowed data for LSTM input.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    window_size (int): Number of time steps in each window.
    features (List[str]): List of features to be used in the model.

    Returns:
    Tuple[np.ndarray, np.ndarray]: X (features) and y (targets) for the LSTM model.
    """
    X, y = [], []
    
    for _, group in df.groupby(['busstop_name', 'cycle_name']):
        # Ensure that busstop_name, cycle_name, and VehSpd_Cval_CPC are not in the features
        relevant_features = [feature for feature in features if feature not in ['busstop_name', 'cycle_name', 'VehSpd_Cval_CPC']]
        
        # Convert timestamp index to a column and include it as a feature
        group_features = group[relevant_features].copy()
        group_features['timestamp'] = group.index
        
        target = group['VehSpd_Cval_CPC']
        timestamps = group.index  # Timestamps are the index
        
        for i in range(len(group) - window_size):
            window_timestamps = timestamps[i:i + window_size + 1]
            if is_continuous(window_timestamps):
                X.append(group_features.iloc[i:i + window_size].values)
                y.append(target.iloc[i + window_size])
            #else:
            #    print(f"Non-continuous timestamps found in window starting at {window_timestamps[0]}")
                
    return np.array(X), np.array(y)

def preprocess_data(df: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the dataframe into windowed features and targets.

    Parameters:
    df (pd.DataFrame): The raw input dataframe.
    config (dict): Configuration dictionary containing model parameters.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Processed features and labels for model training.
    """
    model_config = config['model_config']
    window_size = model_config.get('window_size', 5)
    features = model_config.get('features', [])
    
    X, y = create_windows(df, window_size, features)
    
    print(f"Shape of input features: {X.shape}")
    print(f"Shape of output labels: {y.shape}")
    return X, y
