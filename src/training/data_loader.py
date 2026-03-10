import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict, Optional
from tqdm import tqdm

from ..utils.noise import add_noise


def build_features_targets_avg(
    df: pd.DataFrame,
    target_name: str,
    target_column: str,
    convert_to_radians: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Build features and targets for Experiment I (averaged DPA)."""
    features = df[['r', 'Period', 'DPA']].values.astype(np.float32)
    
    targets = df[target_column].values.astype(np.float32)
    if convert_to_radians:
        targets = np.deg2rad(targets)
    
    return features, targets


def build_features_targets_timeseries(
    df: pd.DataFrame,
    target_name: str,
    target_column: str,
    num_samples: int = 10,
    convert_to_radians: bool = False,
    half_orbit: bool = False,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build features and targets for time series experiments (II-V) from precomputed orbit data."""
    if random_seed is not None:
        np.random.seed(random_seed)
    
    dpa_cols = [f'DPA_{i/10:.1f}' for i in range(1, 11)]
    
    features_list = []
    targets_list = []
    
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Building features for {target_name}",
        leave=False
    ):
        dpa_samples = row[dpa_cols].to_numpy()
        
        if half_orbit:
            start_idx = np.random.randint(0, 6)
            samples_subset = dpa_samples[start_idx:start_idx+5]
        else:
            samples_subset = dpa_samples
        
        feature_row = np.concatenate(([row['r'], row['Period']], samples_subset))
        features_list.append(feature_row)
        
        if target_name == 'spin':
            targets_list.append(row['a'])
        elif target_name == 'incl':
            targets_list.append(np.deg2rad(row['i']) if convert_to_radians else row['i'])
        elif target_name == 'theta':
            targets_list.append(np.deg2rad(row['theta']) if convert_to_radians else row['theta'])
        elif target_name == 'z':
            targets_list.append(row['r'] * np.sin(np.deg2rad(row['theta'])))
    
    features = np.array(features_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32)
    
    metadata = {'num_curves': len(features_list)}
    
    return features, targets, metadata


def prepare_dataloaders(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    noise_enabled: bool = False,
    sigma_r: float = 0.1,
    sigma_T: float = 2.0,
    sigma_DPA: float = 5.0,
    dpa_length_scale: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler, Tuple]:
    """Prepare train/val/test dataloaders with optional noise injection."""
    np.random.seed(random_seed)
    
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    n = len(indices)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    if noise_enabled:
        features_noisy = add_noise(features, sigma_r, sigma_T, sigma_DPA, dpa_length_scale)
    else:
        features_noisy = features.copy()
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(features_noisy[train_idx])
    X_val = scaler_X.transform(features_noisy[val_idx])
    X_test = scaler_X.transform(features_noisy[test_idx])
    
    y_train = scaler_y.fit_transform(targets[train_idx].reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(targets[val_idx].reshape(-1, 1)).ravel()
    y_test = scaler_y.transform(targets[test_idx].reshape(-1, 1)).ravel()
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader, scaler_X, scaler_y, (train_idx, val_idx, test_idx)
