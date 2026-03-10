#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import torch
import wandb

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.config import load_config, get_repo_root
from src.models.regression_head import RegressionHead
from src.training.data_loader import build_features_targets_avg, prepare_dataloaders
from src.training.trainer import Trainer
from src.training.evaluation import (
    evaluate_model,
    plot_error_histogram,
    plot_pred_vs_actual,
    aggregate_results,
    save_results_csv
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
    
    results_dir = root / "results"
    exp_name = config['experiment']['name']
    
    for target_config in config['targets']:
        target_name = target_config['name']
        target_column = target_config['column']
        convert_to_radians = target_config['convert_to_radians']
        
        print(f"\n{'='*60}")
        print(f"Training for target: {target_name}")
        print(f"{'='*60}")
        
        features, targets = build_features_targets_avg(
            df, target_name, target_column, convert_to_radians
        )
        print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")
        
        all_results = []
        
        for seed in config['training']['seeds']:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            
            if config['training']['use_wandb']:
                wandb.init(
                    project=config["training"].get("wandb_project", "smbh-hotspots"),
                    name=f"{exp_name}_{target_name}_seed{seed}",
                    config=config,
                    reinit=True
                )
            
            train_loader, val_loader, test_loader, scaler_X, scaler_y, split_idx = prepare_dataloaders(
                features, targets,
                batch_size=config['training']['batch_size'],
                train_ratio=config['split']['train'],
                val_ratio=config['split']['val'],
                random_seed=seed,
                noise_enabled=config['noise']['enabled'],
                sigma_r=config['noise']['sigma_r'],
                sigma_T=config['noise']['sigma_T'],
                sigma_DPA=config['noise']['sigma_DPA']
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
            
            train_info = trainer.train(
                epochs=config['training']['epochs'],
                use_wandb=config['training']['use_wandb'],
                verbose=True
            )
            
            test_metrics, test_targets, test_preds = evaluate_model(
                model, test_loader, scaler_y, device
            )
            
            print(f"Test MAE: {test_metrics['mae']:.4f}")
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"Test R²: {test_metrics['r2']:.4f}")
            print(f"Error μ: {test_metrics['error_mean']:.4f}, σ: {test_metrics['error_std']:.4f}")
            
            result_row = {
                'target': target_name,
                'seed': seed,
                **test_metrics
            }
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
            
            ckpt_dir = results_dir / "checkpoints" / exp_name / target_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(
                str(ckpt_dir / f"model_seed{seed}.pth"),
                scaler_X_mean=scaler_X.mean_,
                scaler_X_scale=scaler_X.scale_,
                scaler_y_mean=scaler_y.mean_,
                scaler_y_scale=scaler_y.scale_
            )
            
            if config['training']['use_wandb']:
                wandb.finish()
        
        metrics_dir = results_dir / "metrics" / exp_name
        metrics_dir.mkdir(parents=True, exist_ok=True)
        save_results_csv(
            all_results,
            str(metrics_dir / f"{target_name}_summary.csv")
        )
        
        aggregated = aggregate_results(all_results, target_name)
        print(f"\n{'='*60}")
        print(f"Aggregated Results for {target_name}:")
        for key, val in aggregated.items():
            print(f"  {key}: {val:.4f}")
        print(f"{'='*60}")
        
        agg_df = pd.DataFrame([{'target': target_name, **aggregated}])
        agg_df.to_csv(str(metrics_dir / f"{target_name}_aggregated.csv"), index=False)
    
    print("\nExperiment 1 complete!")


if __name__ == "__main__":
    main()
