#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import wandb

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


def build_features_partial_orbit(df, target_name, target_column, num_samples, convert_to_radians, random_seed):
    """Build features with variable number of consecutive DPA samples from precomputed orbit data (wide format)."""
    np.random.seed(random_seed)
    
    dpa_cols = [f'DPA_{i/10:.1f}' for i in range(1, 11)]
    
    features_list = []
    targets_list = []
    
    for _, row in df.iterrows():
        dpa_samples = row[dpa_cols].to_numpy()
        
        start_idx = np.random.randint(0, max(1, 11 - num_samples))
        samples_subset = dpa_samples[start_idx:start_idx+num_samples]
        
        feature_row = np.concatenate(([row['r'], row['Period']], samples_subset))
        features_list.append(feature_row)
        
        if target_name == 'spin':
            targets_list.append(row['a'])
        elif target_name == 'incl':
            targets_list.append(np.deg2rad(row['i']) if convert_to_radians else row['i'])
    
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
    
    required_cols = ['r', 'K', 'a', 'i', 'Period'] + [f'DPA_{i/10:.1f}' for i in range(1, 11)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n{'='*60}\nERROR: Dataset has wrong format!\nMissing columns: {missing_cols[:5]}...")
        print("\nRegenerate with: cd src/preprocessing && python prepare_dataset_ultradense.py")
        print(f"Expected DPA_0.1 ... DPA_1.0 columns. Got: {list(df.columns)[:8]}...\n{'='*60}\n")
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
        wandb_project = config['training'].get('wandb_project', 'exp III')
        
        if config.get('sweep', {}).get('enabled', False):
            print("\n*** Running σ vs % orbit inclusion sweep ***")
            sweep_results = []
            
            for num_samples in config['sweep']['num_samples_range']:
                percent = num_samples * 10
                print(f"\n--- {percent}% orbit ({num_samples} samples) ---")
                
                sigma_values = []
                
                for seed in config['training']['seeds']:
                    torch.manual_seed(seed)
                    
                    if use_wandb:
                        wandb.init(
                            project=wandb_project,
                            name=f"{exp_name}_{target_name}_sweep{percent}_seed{seed}",
                            config=config,
                            reinit=True,
                        )
                    
                    features, targets = build_features_partial_orbit(
                        df, target_name, target_column, num_samples,
                        convert_to_radians, seed
                    )
                    
                    train_loader, val_loader, test_loader, scaler_X, scaler_y, _ = prepare_dataloaders(
                        features, targets,
                        batch_size=config['training']['batch_size'],
                        train_ratio=config['split']['train'],
                        val_ratio=config['split']['val'],
                        random_seed=seed,
                        noise_config=config['noise']
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
                    
                    trainer.train(
                        epochs=config['training']['epochs'],
                        use_wandb=use_wandb,
                        verbose=False,
                    )
                    
                    if use_wandb:
                        wandb.finish()
                    
                    test_metrics, _, _ = evaluate_model(model, test_loader, scaler_y, device)
                    sigma_values.append(test_metrics['error_std'])
                
                mean_sigma = np.mean(sigma_values)
                std_sigma = np.std(sigma_values, ddof=1)
                print(f"  σ = {mean_sigma:.4f} ± {std_sigma:.4f}")
                
                sweep_results.append({
                    'percent': percent,
                    'num_samples': num_samples,
                    'sigma_mean': mean_sigma,
                    'sigma_std': std_sigma
                })
            
            metrics_dir = results_dir / "metrics" / exp_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            sweep_df = pd.DataFrame(sweep_results)
            sweep_df.to_csv(str(metrics_dir / f"{target_name}_sweep.csv"), index=False)
            
            fig_dir = results_dir / "figures" / exp_name
            fig_dir.mkdir(parents=True, exist_ok=True)
            plot_sigma_vs_orbit_inclusion(
                [r['sigma_mean'] for r in sweep_results],
                [r['percent'] for r in sweep_results],
                target_name,
                str(fig_dir / f"{target_name}_sigma_vs_orbit.png")
            )
        
        else:
            print("\n*** Running standard half-orbit training ***")
            
            features, targets = build_features_partial_orbit(
                df, target_name, target_column, 5,
                convert_to_radians, config['split']['random_seed']
            )
            print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
            
            all_results = []
            
            for seed in config['training']['seeds']:
                print(f"\n--- Seed {seed} ---")
                torch.manual_seed(seed)
                
                if use_wandb:
                    wandb.init(
                        project=wandb_project,
                        name=f"{exp_name}_{target_name}_seed{seed}",
                        config=config,
                        reinit=True,
                    )
                
                train_loader, val_loader, test_loader, scaler_X, scaler_y, _ = prepare_dataloaders(
                    features, targets,
                    batch_size=config['training']['batch_size'],
                    train_ratio=config['split']['train'],
                    val_ratio=config['split']['val'],
                    random_seed=seed,
                    noise_config=config['noise']
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
                
                trainer.train(
                    epochs=config['training']['epochs'],
                    use_wandb=use_wandb,
                    verbose=True,
                )
                
                test_metrics, test_targets, test_preds = evaluate_model(
                    model, test_loader, scaler_y, device
                )
                
                print(f"Test MAE: {test_metrics['mae']:.4f}")
                print(f"Test R²: {test_metrics['r2']:.4f}")
                print(f"Error σ: {test_metrics['error_std']:.4f}")
                
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
    
    print("\nExperiment 3 complete!")


if __name__ == "__main__":
    main()
