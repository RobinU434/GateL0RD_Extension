#!/usr/bin/env python3
"""
Evaluation script for GateL0RD models.

This script provides comprehensive evaluation capabilities including:
- Model performance evaluation on test sets
- Architecture comparison across versions
- Parameter efficiency analysis
- Visualization of results

Usage:
    python evaluate.py --checkpoint path/to/model.ckpt --data_dir data/BilliardBall
    python evaluate.py --experiment_dir logs/comparison --compare_versions
    python evaluate.py --config config/evaluation.yaml --output_dir results/
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from project.utils.config import ConfigManager, ExperimentConfig, load_config_from_args
from project.gatel0rd.lightning_module import GateL0RDLightningModule
from project.datasets.timeseries import TimeSeriesDataModule


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def load_model_from_checkpoint(self, checkpoint_path: str) -> GateL0RDLightningModule:
        """Load trained model from checkpoint."""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        model = GateL0RDLightningModule.load_from_checkpoint(checkpoint_path)
        model.eval()
        
        return model
        
    def evaluate_model_performance(
        self, 
        model: GateL0RDLightningModule, 
        data_module: TimeSeriesDataModule,
        dataset_split: str = 'test'
    ) -> Dict[str, Any]:
        """Evaluate model performance on dataset."""
        self.logger.info(f"Evaluating model on {dataset_split} set")
        
        # Setup data
        data_module.setup('test' if dataset_split == 'test' else 'fit')
        
        if dataset_split == 'test':
            dataloader = data_module.test_dataloader()
        elif dataset_split == 'val':
            dataloader = data_module.val_dataloader()
        else:
            dataloader = data_module.train_dataloader()
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_theta_stats = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_seq = batch['input']
                target_seq = batch['target']
                
                # Forward pass
                outputs, final_hidden, theta = model(input_seq)
                
                # Get predictions (last prediction_steps outputs)
                predictions = outputs[:, -model.prediction_steps:, :]
                
                if model.prediction_steps == 1:
                    predictions = predictions.squeeze(1)
                    target_seq = target_seq.squeeze(1)
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target_seq.cpu().numpy())
                
                # Theta statistics
                all_theta_stats.append({
                    'mean': theta.mean().item(),
                    'std': theta.std().item(), 
                    'sparsity': (theta < 0.5).float().mean().item(),
                    'activation_rate': (theta > 0.5).float().mean().item(),
                })
        
        # Combine all batches
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate metrics
        if predictions.ndim > 2:
            # Multi-step prediction - flatten for metrics
            predictions = predictions.reshape(-1, predictions.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
            
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # Per-feature metrics
        n_features = targets.shape[-1]
        for i in range(n_features):
            feature_targets = targets[:, i]
            feature_predictions = predictions[:, i]
            
            metrics[f'feature_{i}_mse'] = mean_squared_error(feature_targets, feature_predictions)
            metrics[f'feature_{i}_mae'] = mean_absolute_error(feature_targets, feature_predictions)
            metrics[f'feature_{i}_r2'] = r2_score(feature_targets, feature_predictions)
        
        # Theta statistics
        theta_stats = pd.DataFrame(all_theta_stats)
        metrics['theta_mean_avg'] = theta_stats['mean'].mean()
        metrics['theta_std_avg'] = theta_stats['std'].mean()
        metrics['theta_sparsity_avg'] = theta_stats['sparsity'].mean()
        metrics['theta_activation_rate_avg'] = theta_stats['activation_rate'].mean()
        
        # Additional analysis
        metrics['prediction_variance'] = np.var(predictions, axis=0).mean()
        metrics['target_variance'] = np.var(targets, axis=0).mean()
        metrics['explained_variance'] = 1 - (np.var(targets - predictions, axis=0) / np.var(targets, axis=0)).mean()
        
        self.logger.info(f"Evaluation completed. MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'theta_stats': all_theta_stats,
        }
    
    def compare_model_versions(
        self, 
        checkpoint_paths: Dict[str, str],
        data_module: TimeSeriesDataModule
    ) -> Dict[str, Any]:
        """Compare multiple model versions."""
        self.logger.info("Comparing model versions")
        
        comparison_results = {}
        
        for version, checkpoint_path in checkpoint_paths.items():
            self.logger.info(f"Evaluating version {version}")
            
            try:
                model = self.load_model_from_checkpoint(checkpoint_path)
                results = self.evaluate_model_performance(model, data_module)
                
                # Add model complexity info
                complexity = model.get_model_complexity()
                results['complexity'] = complexity
                results['model_version'] = version
                
                comparison_results[version] = results
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {version}: {str(e)}")
                comparison_results[version] = {'error': str(e)}
        
        # Create comparison summary
        summary = self._create_comparison_summary(comparison_results)
        
        return {
            'individual_results': comparison_results,
            'summary': summary,
        }
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary comparison of model versions."""
        summary = {
            'performance_ranking': [],
            'efficiency_ranking': [],
            'parameter_reduction': {},
            'best_performers': {},
        }
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            return summary
        
        # Performance ranking (by R²)
        performance_scores = [(version, data['metrics']['r2']) for version, data in valid_results.items()]
        performance_scores.sort(key=lambda x: x[1], reverse=True)
        summary['performance_ranking'] = performance_scores
        
        # Efficiency ranking (performance per parameter)
        efficiency_scores = []
        for version, data in valid_results.items():
            r2_score = data['metrics']['r2']
            total_params = data['complexity']['total_params']
            efficiency = r2_score / (total_params / 1000)  # R² per 1K parameters
            efficiency_scores.append((version, efficiency))
        
        efficiency_scores.sort(key=lambda x: x[1], reverse=True)
        summary['efficiency_ranking'] = efficiency_scores
        
        # Parameter reduction compared to v0
        if 'v0' in valid_results:
            v0_params = valid_results['v0']['complexity']['total_params']
            for version, data in valid_results.items():
                if version != 'v0':
                    version_params = data['complexity']['total_params']
                    reduction = (v0_params - version_params) / v0_params
                    summary['parameter_reduction'][version] = reduction
        
        # Best performers by metric
        metrics_to_compare = ['mse', 'mae', 'r2', 'theta_sparsity_avg']
        for metric in metrics_to_compare:
            if metric == 'r2':
                best = max(valid_results.items(), key=lambda x: x[1]['metrics'][metric])
            else:
                best = min(valid_results.items(), key=lambda x: x[1]['metrics'][metric])
            summary['best_performers'][metric] = best[0]
        
        return summary
    
    def analyze_parameter_efficiency(
        self, 
        results: Dict[str, Any],
        baseline_version: str = 'v0'
    ) -> Dict[str, Any]:
        """Analyze parameter efficiency across versions."""
        if baseline_version not in results['individual_results']:
            raise ValueError(f"Baseline version {baseline_version} not found in results")
        
        baseline_results = results['individual_results'][baseline_version]
        baseline_performance = baseline_results['metrics']['r2']
        baseline_params = baseline_results['complexity']['total_params']
        
        efficiency_analysis = {
            'baseline_version': baseline_version,
            'baseline_performance': baseline_performance,
            'baseline_params': baseline_params,
            'efficiency_ratios': {},
            'pareto_efficient': [],
        }
        
        for version, data in results['individual_results'].items():
            if 'error' in data or version == baseline_version:
                continue
                
            performance = data['metrics']['r2']
            params = data['complexity']['total_params']
            
            # Calculate efficiency metrics
            performance_ratio = performance / baseline_performance
            param_ratio = params / baseline_params
            efficiency_ratio = performance_ratio / param_ratio
            
            efficiency_analysis['efficiency_ratios'][version] = {
                'performance_ratio': performance_ratio,
                'parameter_ratio': param_ratio,
                'efficiency_ratio': efficiency_ratio,
                'parameter_reduction': 1 - param_ratio,
                'performance_retention': performance_ratio,
            }
            
            # Check if Pareto efficient (better performance with fewer parameters)
            if performance >= baseline_performance * 0.95 and params < baseline_params:
                efficiency_analysis['pareto_efficient'].append(version)
        
        return efficiency_analysis


class ResultVisualizer:
    """Visualization utilities for evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_performance_comparison(self, comparison_results: Dict[str, Any]):
        """Plot performance comparison across versions."""
        results = comparison_results['individual_results']
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if len(valid_results) < 2:
            return
        
        # Prepare data
        versions = list(valid_results.keys())
        metrics = ['mse', 'mae', 'r2', 'theta_sparsity_avg']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [valid_results[v]['metrics'][metric] for v in versions]
            
            ax = axes[i]
            bars = ax.bar(versions, values, alpha=0.7)
            ax.set_title(f'{metric.upper().replace("_", " ")}')
            ax.set_xlabel('Model Version')
            ax.set_ylabel(metric.upper())
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_parameter_efficiency(self, efficiency_analysis: Dict[str, Any]):
        """Plot parameter efficiency analysis."""
        ratios = efficiency_analysis['efficiency_ratios']
        
        if not ratios:
            return
        
        versions = list(ratios.keys())
        performance_ratios = [ratios[v]['performance_ratio'] for v in versions]
        parameter_ratios = [ratios[v]['parameter_ratio'] for v in versions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: Performance vs Parameters
        colors = ['red' if v in efficiency_analysis['pareto_efficient'] else 'blue' for v in versions]
        
        ax1.scatter(parameter_ratios, performance_ratios, c=colors, s=100, alpha=0.7)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Performance')
        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Parameters')
        ax1.axhline(y=0.95, color='orange', linestyle=':', alpha=0.5, label='95% Performance')
        
        for i, version in enumerate(versions):
            ax1.annotate(version, (parameter_ratios[i], performance_ratios[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Parameter Ratio (vs Baseline)')
        ax1.set_ylabel('Performance Ratio (vs Baseline)')
        ax1.set_title('Parameter Efficiency Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bar plot: Efficiency ratios
        efficiency_ratios = [ratios[v]['efficiency_ratio'] for v in versions]
        bars = ax2.bar(versions, efficiency_ratios, alpha=0.7)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline Efficiency')
        ax2.set_title('Efficiency Ratio (Performance/Parameters)')
        ax2.set_xlabel('Model Version')
        ax2.set_ylabel('Efficiency Ratio')
        ax2.legend()
        
        # Color bars based on efficiency
        for bar, ratio in zip(bars, efficiency_ratios):
            if ratio > 1.1:
                bar.set_color('green')
            elif ratio > 0.9:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_efficiency.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_complexity_breakdown(self, comparison_results: Dict[str, Any]):
        """Plot model complexity breakdown by components."""
        results = comparison_results['individual_results']
        valid_results = {k: v for k, v in results.items() if 'error' not in v and 'complexity' in v}
        
        if len(valid_results) < 2:
            return
        
        versions = list(valid_results.keys())
        components = ['g_network_params', 'r_network_params', 'output_network_params']
        
        # Prepare data for stacked bar chart
        data = {component: [valid_results[v]['complexity'][component] for v in versions] 
               for component in components}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bottom = np.zeros(len(versions))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, component in enumerate(components):
            values = data[component]
            ax.bar(versions, values, bottom=bottom, label=component.replace('_', ' ').title(), 
                  color=colors[i], alpha=0.8)
            bottom += values
        
        ax.set_title('Model Complexity Breakdown by Component')
        ax.set_xlabel('Model Version')
        ax.set_ylabel('Number of Parameters')
        ax.legend()
        
        # Add total parameter counts on top of bars
        for i, version in enumerate(versions):
            total = valid_results[version]['complexity']['total_params']
            ax.text(i, total + max(bottom) * 0.01, f'{total:,}', 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'complexity_breakdown.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate GateL0RD models")
    
    # Input options
    parser.add_argument('--checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--experiment_dir', type=str, help="Path to experiment directory")
    parser.add_argument('--config', type=str, help="Path to evaluation config")
    
    # Data options  
    parser.add_argument('--data_dir', type=str, help="Path to data directory")
    parser.add_argument('--dataset_name', type=str, default='BilliardBall')
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Evaluation options
    parser.add_argument('--compare_versions', action='store_true', help="Compare all versions")
    parser.add_argument('--dataset_split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--baseline_version', default='v0', help="Baseline for comparison")
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--save_predictions', action='store_true', help="Save predictions")
    parser.add_argument('--create_plots', action='store_true', help="Create visualization plots")
    
    # Logging
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(logger)
    
    try:
        if args.compare_versions and args.experiment_dir:
            # Compare versions from experiment directory
            logger.info("Running version comparison evaluation")
            
            experiment_dir = Path(args.experiment_dir)
            checkpoint_paths = {}
            
            # Find checkpoints for each version
            for version in ['v0', 'v1', 'v2', 'v3']:
                version_checkpoints = list((experiment_dir / 'checkpoints' / version).glob('*.ckpt'))
                if version_checkpoints:
                    # Use the best checkpoint (assumes filename contains val_loss)
                    best_checkpoint = min(version_checkpoints, 
                                        key=lambda x: float(str(x.stem).split('val_loss-')[-1].split('-')[0]))
                    checkpoint_paths[version] = str(best_checkpoint)
            
            if not checkpoint_paths:
                raise ValueError(f"No checkpoints found in {experiment_dir}")
            
            logger.info(f"Found checkpoints for versions: {list(checkpoint_paths.keys())}")
            
            # Setup data module
            data_module = TimeSeriesDataModule(
                data_dir=args.data_dir or str(experiment_dir.parent / 'data' / args.dataset_name),
                batch_size=args.batch_size,
            )
            
            # Run comparison
            results = evaluator.compare_model_versions(checkpoint_paths, data_module)
            
            # Analyze efficiency
            efficiency_analysis = evaluator.analyze_parameter_efficiency(results, args.baseline_version)
            
            # Save results
            with open(output_dir / 'comparison_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            with open(output_dir / 'efficiency_analysis.json', 'w') as f:
                json.dump(efficiency_analysis, f, indent=2, default=str)
            
            # Create visualizations
            if args.create_plots:
                visualizer = ResultVisualizer(str(output_dir))
                visualizer.plot_performance_comparison(results)
                visualizer.plot_parameter_efficiency(efficiency_analysis)
                visualizer.plot_complexity_breakdown(results)
            
            # Print summary
            logger.info("=== Comparison Summary ===")
            summary = results['summary']
            logger.info(f"Performance ranking: {summary['performance_ranking']}")
            logger.info(f"Efficiency ranking: {summary['efficiency_ranking']}")
            logger.info(f"Pareto efficient models: {efficiency_analysis['pareto_efficient']}")
            
        elif args.checkpoint:
            # Single model evaluation
            logger.info("Running single model evaluation")
            
            # Setup data module
            data_module = TimeSeriesDataModule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
            )
            
            # Load and evaluate model
            model = evaluator.load_model_from_checkpoint(args.checkpoint)
            results = evaluator.evaluate_model_performance(model, data_module, args.dataset_split)
            
            # Save results
            with open(output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(results['metrics'], f, indent=2, default=str)
            
            if args.save_predictions:
                np.save(output_dir / 'predictions.npy', results['predictions'])
                np.save(output_dir / 'targets.npy', results['targets'])
            
            # Print summary
            logger.info("=== Evaluation Results ===")
            metrics = results['metrics']
            logger.info(f"MSE: {metrics['mse']:.6f}")
            logger.info(f"RMSE: {metrics['rmse']:.6f}")
            logger.info(f"MAE: {metrics['mae']:.6f}")
            logger.info(f"R²: {metrics['r2']:.6f}")
            logger.info(f"Theta sparsity: {metrics['theta_sparsity_avg']:.3f}")
            
        else:
            raise ValueError("Must provide either --checkpoint or --experiment_dir with --compare_versions")
        
        logger.info(f"Evaluation completed. Results saved to {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())