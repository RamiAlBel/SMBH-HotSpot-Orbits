import numpy as np


def rbf_kernel(t1: np.ndarray, t2: np.ndarray, sigma: float, length_scale: float) -> np.ndarray:
    """Compute RBF (Squared Exponential) kernel covariance matrix.
    
    K(t_i, t_j) = sigma^2 * exp(- (t_i - t_j)^2 / (2 * length_scale^2))
    
    Args:
        t1: Array of time points (shape: n1,)
        t2: Array of time points (shape: n2,)
        sigma: Amplitude of the noise
        length_scale: Controls smoothness (larger = smoother)
    
    Returns:
        Covariance matrix of shape (n1, n2)
    """
    t1 = t1.reshape(-1, 1)
    t2 = t2.reshape(1, -1)
    sq_dist = (t1 - t2) ** 2
    return sigma**2 * np.exp(-sq_dist / (2 * length_scale**2))


def sample_gp_noise(time_points: np.ndarray, sigma: float, length_scale: float) -> np.ndarray:
    """Sample smooth noise from a Gaussian Process with RBF kernel.

    The RBF kernel is parameterised so that K[i,i] = sigma^2, guaranteeing
    that every time point has the correct marginal std dev (sigma) regardless
    of how many points are sampled.  No empirical rescaling is applied because
    it is both unnecessary and numerically unstable for small k (e.g. k=2).

    Args:
        time_points: Array of time points (e.g., [0.1, 0.2, ..., 1.0])
        sigma: Target std dev of the noise at each time point
        length_scale: Smoothness parameter (larger = smoother)

    Returns:
        Array of correlated noise values; each element has marginal std = sigma
    """
    K = rbf_kernel(time_points, time_points, sigma, length_scale)

    # Add small jitter for numerical stability
    K += 1e-8 * np.eye(len(time_points))

    return np.random.multivariate_normal(np.zeros(len(time_points)), K)


def add_noise(
    features: np.ndarray,
    sigma_r: float,
    sigma_T: float,
    sigma_DPA: float,
    dpa_length_scale: float = 0.0
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
        dpa_length_scale: Length scale for GP-based DPA noise.
                         If 0, uses independent Gaussian noise (old behavior).
                         If > 0, uses smooth GP noise with RBF kernel.
    
    Returns:
        Noisy features array
    """
    noisy = features.copy()
    n_samples = noisy.shape[0]
    
    noisy[:, 0] += np.random.normal(0, sigma_r, size=n_samples)
    noisy[:, 1] += np.random.normal(0, sigma_T, size=n_samples)
    
    if noisy.shape[1] > 2:
        n_dpa_cols = noisy.shape[1] - 2
        
        if dpa_length_scale > 0:
            # GP-based smooth noise
            time_points = np.linspace(0.1, 1.0, n_dpa_cols)
            
            for i in range(n_samples):
                gp_noise = sample_gp_noise(time_points, sigma_DPA, dpa_length_scale)
                noisy[i, 2:] += gp_noise
        else:
            # Independent Gaussian noise (old behavior)
            dpa_shape = noisy[:, 2:].shape
            noisy[:, 2:] += np.random.normal(0, sigma_DPA, size=dpa_shape)
    
    return noisy
