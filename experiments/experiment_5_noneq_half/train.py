#!/usr/bin/env python3
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import torch
import wandb
from tqdm import tqdm

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.config import load_config, get_repo_root
from src.models.regression_head import RegressionHead
from src.training.trainer import Trainer
from src.training.evaluation import (
    evaluate_model,
    plot_error_histogram,
    plot_pred_vs_actual,
    aggregate_results,
    save_results_csv,
    plot_sigma_vs_orbit_inclusion
)


def build_features_partial_orbit_noneq(df, target_name, target_column, num_samples, convert_to_radians, random_seed):
    """Build features with variable number of consecutive DPA samples from precomputed orbit data."""
    np.random.seed(random_seed)
    
    dpa_cols = [f'DPA_{i/10:.1f}' for i in range(1, 11)]
    
    features_list = []
    targets_list = []
    
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{target_name}: building features",
        leave=False,
    ):
        dpa_samples = row[dpa_cols].to_numpy()
        
        start_idx = np.random.randint(0, max(1, 11 - num_samples))
        samples_subset = dpa_samples[start_idx:start_idx+num_samples]
        
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
    
    return features, targets


def prepare_dataloaders(features, targets, batch_size, train_ratio, val_ratio, random_seed, noise_config):
    """Prepare dataloaders with noise injection."""
    from src.training.data_loader import prepare_dataloaders as prep_dl
    return prep_dl(
        features, targets, batch_size, train_ratio, val_ratio, random_seed,
        noise_config['enabled'], noise_config['sigma_r'], 
        noise_config['sigma_T'], noise_config['sigma_DPA'],
        noise_config.get('dpa_length_scale', 0.0)
    )


def main():
    config_name = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config_path = Path(__file__).parent / config_name
    print(f"Loading config: {config_name}")
    config = load_config(config_path)
    
    root = get_repo_root()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    dataset_path = root / config['data']['dataset_path']
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} rows")
    
    required_cols = ['r', 'K', 'a', 'i', 'theta', 'Period'] + [f'DPA_{i/10:.1f}' for i in range(1, 11)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n{'='*60}")
        print("ERROR: Dataset has wrong format!")
        print(f"Missing columns: {missing_cols[:5]}...")
        print("\nYou need to regenerate the dataset with:")
        print("  cd src/preprocessing")
        print("  python prepare_dataset_noneq.py")
        print("\nExpected ~5,000 rows with DPA_0.1, DPA_0.2, ..., DPA_1.0 columns")
        print(f"Got {len(df)} rows with columns: {list(df.columns)[:10]}...")
        print(f"{'='*60}\n")
        sys.exit(1)
    
    results_dir = root / "results"
    exp_name = config['experiment']['name']
    
    for target_config in config['targets']:
        target_name = target_config['name']
        target_column = target_config['column']
        convert_to_radians = target_config['convert_to_radians']
        
        print(f"\n{'='*60}")
        print(f"Training for target: {target_name}")
        print(f"{'='*60}")
        
        use_wandb = config['training'].get('use_wandb', False)
        wandb_project = config['training'].get('wandb_project', 'exp V')
        
        if config.get('sweep', {}).get('enabled', False):
            print("\n*** Running σ vs % orbit inclusion sweep ***")
            sweep_results = []
            
            for num_samples in config['sweep']['num_samples_range']:
                percent = num_samples * 10
                print(f"\n--- {percent}% orbit ({num_samples} samples) ---")
                
                seed_metrics = []
                
                for seed in tqdm(
                    config['training']['seeds'],
                    desc=f"{target_name}: seeds (sweep, {percent}%)",
                    leave=False,
                ):
                    seed_start = time.time()
                    torch.manual_seed(seed)
                    
                    if use_wandb:
                        wandb.init(
                            project=wandb_project,
                            name=f"{exp_name}_{target_name}_sweep{percent}_seed{seed}",
                            config=config,
                            reinit=True,
                        )
                    
                    t0 = time.time()
                    features, targets = build_features_partial_orbit_noneq(
                        df, target_name, target_column, num_samples,
                        convert_to_radians, seed
                    )
                    
                    # Check for NaN/Inf
                    nan_features = np.isnan(features).any(axis=1)
                    inf_features = np.isinf(features).any(axis=1)
                    nan_targets = np.isnan(targets)
                    inf_targets = np.isinf(targets)
                    
                    if nan_features.any() or inf_features.any() or nan_targets.any() or inf_targets.any():
                        bad_rows = nan_features | inf_features | nan_targets | inf_targets
                        print(f"[WARNING] Removing {bad_rows.sum()} bad rows with NaN/Inf", flush=True)
                        features = features[~bad_rows]
                        targets = targets[~bad_rows]
                    
                    print(
                        f"[{target_name}] sweep {percent}% seed {seed}: "
                        f"build_features {time.time() - t0:.1f}s",
                        flush=True,
                    )
                    
                    t0 = time.time()
                    train_loader, val_loader, test_loader, scaler_X, scaler_y, _ = prepare_dataloaders(
                        features, targets,
                        batch_size=config['training']['batch_size'],
                        train_ratio=config['split']['train'],
                        val_ratio=config['split']['val'],
                        random_seed=seed,
                        noise_config=config['noise']
                    )
                    print(
                        f"[{target_name}] sweep {percent}% seed {seed}: "
                        f"prepare_dataloaders {time.time() - t0:.1f}s",
                        flush=True,
                    )
                    
                    model = RegressionHead(
                        input_dim=features.shape[1],
                        hidden_dims=tuple(config['model']['hidden_dims']),
                        num_blocks=config['model']['num_blocks'],
                        dropout=config['model']['dropout']
                    )
                    
                    trainer = Trainer(
                        model, train_loader, val_loader,
                        learning_rate=config['training']['learning_rate'],
                        weight_decay=config['training']['weight_decay'],
                        early_stop_patience=config['training']['early_stop_patience'],
                        device=device
                    )
                    
                    t0 = time.time()
                    trainer.train(
                        epochs=config['training']['epochs'],
                        use_wandb=use_wandb,
                        verbose=False,
                    )
                    print(
                        f"[{target_name}] sweep {percent}% seed {seed}: "
                        f"train {time.time() - t0:.1f}s",
                        flush=True,
                    )
                    
                    t0 = time.time()
                    test_metrics, _, _ = evaluate_model(model, test_loader, scaler_y, device)
                    eval_time = time.time() - t0
                    seed_metrics.append(test_metrics)
                    
                    print(
                        f"[{target_name}] sweep {percent}% seed {seed}: "
                        f"eval {eval_time:.1f}s, total {time.time() - seed_start:.1f}s",
                        flush=True,
                    )
                    
                    if use_wandb:
                        wandb.finish()
                
                keys = ['mae', 'rmse', 'r2', 'error_std']
                agg = {
                    'percent': percent,
                    'num_samples': num_samples,
                }
                for k in keys:
                    vals = [m[k] for m in seed_metrics]
                    agg[f'{k}_mean'] = np.mean(vals)
                    agg[f'{k}_std'] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                print(f"  MAE = {agg['mae_mean']:.4f}, RMSE = {agg['rmse_mean']:.4f}, R² = {agg['r2_mean']:.4f}, σ = {agg['error_std_mean']:.4f}")
                sweep_results.append(agg)
            
            metrics_dir = results_dir / "metrics" / exp_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df.to_csv(str(metrics_dir / f"{target_name}_sweep.csv"), index=False)
            
            row_100 = sweep_df[sweep_df['percent'] == 100].iloc[0]
            agg_row = {
                'target': target_name,
                'mae_mean': row_100['mae_mean'], 'mae_std': row_100['mae_std'],
                'rmse_mean': row_100['rmse_mean'], 'rmse_std': row_100['rmse_std'],
                'r2_mean': row_100['r2_mean'], 'r2_std': row_100['r2_std'],
                'error_std_mean': row_100['error_std_mean'],
                'error_std_std': row_100['error_std_std'],
            }
            pd.DataFrame([agg_row]).to_csv(str(metrics_dir / f"{target_name}_aggregated.csv"), index=False)
            
            fig_dir = results_dir / "figures" / exp_name
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_sigma_vs_orbit_inclusion(
                [r['error_std_mean'] for r in sweep_results],
                [r['percent'] for r in sweep_results],
                target_name,
                str(fig_dir / f"{target_name}_sigma_vs_orbit.png")
            )
        
        else:
            print("\n*** Running standard half-orbit training ***")
            
            features, targets = build_features_partial_orbit_noneq(
                df, target_name, target_column, 5,
                convert_to_radians, config['split']['random_seed']
            )
            
            # Check for NaN/Inf
            nan_features = np.isnan(features).any(axis=1)
            inf_features = np.isinf(features).any(axis=1)
            nan_targets = np.isnan(targets)
            inf_targets = np.isinf(targets)
            
            print(f"Features: {nan_features.sum()} rows with NaN, {inf_features.sum()} rows with Inf")
            print(f"Targets: {nan_targets.sum()} NaN values, {inf_targets.sum()} Inf values")
            
            if nan_features.any() or inf_features.any() or nan_targets.any() or inf_targets.any():
                bad_rows = nan_features | inf_features | nan_targets | inf_targets
                print(f"[WARNING] Removing {bad_rows.sum()} bad rows with NaN/Inf")
                features = features[~bad_rows]
                targets = targets[~bad_rows]
            
            print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
            
            all_results = []
            
            for seed in tqdm(
                config['training']['seeds'],
                desc=f"{target_name}: seeds (standard half-orbit)",
                leave=False,
            ):
                print(f"\n--- Seed {seed} ---")
                seed_start = time.time()
                torch.manual_seed(seed)
                
                if use_wandb:
                    wandb.init(
                        project=wandb_project,
                        name=f"{exp_name}_{target_name}_seed{seed}",
                        config=config,
                        reinit=True,
                    )
                
                t0 = time.time()
                train_loader, val_loader, test_loader, scaler_X, scaler_y, _ = prepare_dataloaders(
                    features, targets,
                    batch_size=config['training']['batch_size'],
                    train_ratio=config['split']['train'],
                    val_ratio=config['split']['val'],
                    random_seed=seed,
                    noise_config=config['noise']
                )
                print(
                    f"[{target_name}] seed {seed}: prepare_dataloaders {time.time() - t0:.1f}s",
                    flush=True,
                )
                
                model = RegressionHead(
                    input_dim=features.shape[1],
                    hidden_dims=tuple(config['model']['hidden_dims']),
                    num_blocks=config['model']['num_blocks'],
                    dropout=config['model']['dropout']
                )
                
                trainer = Trainer(
                    model, train_loader, val_loader,
                    learning_rate=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay'],
                    early_stop_patience=config['training']['early_stop_patience'],
                    device=device
                )
                
                t0 = time.time()
                trainer.train(
                    epochs=config['training']['epochs'],
                    use_wandb=use_wandb,
                    verbose=True,
                )
                print(
                    f"[{target_name}] seed {seed}: train {time.time() - t0:.1f}s",
                    flush=True,
                )
                
                t0 = time.time()
                test_metrics, test_targets, test_preds = evaluate_model(
                    model, test_loader, scaler_y, device
                )
                eval_time = time.time() - t0
                
                print(f"Test MAE: {test_metrics['mae']:.4f}")
                print(f"Test R²: {test_metrics['r2']:.4f}")
                print(f"Error σ: {test_metrics['error_std']:.4f}")
                print(
                    f"[{target_name}] seed {seed}: eval {eval_time:.1f}s, "
                    f"total {time.time() - seed_start:.1f}s",
                    flush=True,
                )
                
                if use_wandb:
                    wandb.finish()
                
                result_row = {'target': target_name, 'seed': seed, **test_metrics}
                all_results.append(result_row)
                
                fig_dir = results_dir / "figures" / exp_name / target_name
                fig_dir.mkdir(parents=True, exist_ok=True)
                
                errors = test_targets - test_preds
                plot_error_histogram(
                    errors, target_name,
                    str(fig_dir / f"error_hist_seed{seed}.png")
                )
                plot_pred_vs_actual(
                    test_targets, test_preds, target_name,
                    str(fig_dir / f"pred_vs_actual_seed{seed}.png")
                )
            
            metrics_dir = results_dir / "metrics" / exp_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            save_results_csv(all_results, str(metrics_dir / f"{target_name}_summary.csv"))
            
            aggregated = aggregate_results(all_results, target_name)
            print(f"\nAggregated Results for {target_name}:")
            for key, val in aggregated.items():
                print(f"  {key}: {val:.4f}")
            
            agg_df = pd.DataFrame([{'target': target_name, **aggregated}])
            agg_df.to_csv(str(metrics_dir / f"{target_name}_aggregated.csv"), index=False)
    
    print("\nExperiment 5 complete!")


if __name__ == "__main__":
    main()
