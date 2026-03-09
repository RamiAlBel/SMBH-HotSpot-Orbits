import numpy as np


def add_noise(
    features: np.ndarray,
    sigma_r: float,
    sigma_T: float,
    sigma_DPA: float
) -> np.ndarray:
    """Add Gaussian noise to features before normalization.
    
    Args:
        features: Array of shape (n_samples, n_features) where
                  features[:, 0] = r
                  features[:, 1] = Period (T)
                  features[:, 2:] = DPA samples
        sigma_r: Std dev for r noise (in units of M)
        sigma_T: Std dev for Period noise (in minutes)
        sigma_DPA: Std dev for DPA noise (in degrees)
    
    Returns:
        Noisy features array
    """
    noisy = features.copy()
    n_samples = noisy.shape[0]
    
    noisy[:, 0] += np.random.normal(0, sigma_r, size=n_samples)
    noisy[:, 1] += np.random.normal(0, sigma_T, size=n_samples)
    
    if noisy.shape[1] > 2:
        dpa_shape = noisy[:, 2:].shape
        noisy[:, 2:] += np.random.normal(0, sigma_DPA, size=dpa_shape)
    
    return noisy
