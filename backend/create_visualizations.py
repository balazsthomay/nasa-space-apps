"""
Create comprehensive visualizations for model performance and explainability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import lightgbm as lgb
import xgboost as xgb
import shap

from ensemble_predictor import EnsemblePredictor


class VisualizationSuite:
    """Create visualizations for model analysis."""

    def __init__(self, models_dir: str = "models", output_dir: str = "visualizations"):
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

    def plot_model_comparison(self, metrics_path: str = None):
        """Plot comparison of all model performances."""
        if metrics_path is None:
            metrics_path = self.models_dir / "metrics.json"

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Extract test metrics
        models = []
        accuracy = []
        precision = []
        recall = []
        f1 = []
        auc_scores = []

        for model_name, model_metrics in metrics.items():
            test_metrics = model_metrics['test']
            models.append(model_name.replace('_', ' ').title())
            accuracy.append(test_metrics['accuracy'])
            precision.append(test_metrics['precision'])
            recall.append(test_metrics['recall'])
            f1.append(test_metrics['f1'])
            auc_scores.append(test_metrics['roc_auc'])

        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold')

        metrics_data = [
            ('Accuracy', accuracy),
            ('Precision', precision),
            ('Recall', recall),
            ('F1 Score', f1),
            ('ROC AUC', auc_scores)
        ]

        colors = sns.color_palette("husl", len(models))

        for idx, (metric_name, values) in enumerate(metrics_data):
            ax = axes[idx // 3, idx % 3]

            bars = ax.bar(range(len(models)), values, color=colors)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_ylim([0.8, 1.0])
            ax.set_title(metric_name)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=9)

        # All metrics together
        ax = axes[1, 2]
        x = np.arange(len(models))
        width = 0.15

        ax.bar(x - 2*width, accuracy, width, label='Accuracy', alpha=0.8)
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1', alpha=0.8)
        ax.bar(x + 2*width, auc_scores, width, label='AUC', alpha=0.8)

        ax.set_ylabel('Score')
        ax.set_title('All Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0.8, 1.0])

        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_confusion_matrices(self, predictor: EnsemblePredictor, X_test, y_test):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')

        # Get predictions from ensemble
        results = predictor.predict_ensemble(X_test)

        # Individual models
        for idx, (model_name, pred) in enumerate(results['individual_predictions'].items()):
            ax = axes[idx // 3, idx % 3]

            cm = confusion_matrix(y_test, pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Planet', 'Planet'],
                       yticklabels=['Not Planet', 'Planet'])

            ax.set_title(model_name.replace('_', ' ').title())
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

        # Ensemble
        ax = axes[1, 2]
        cm = confusion_matrix(y_test, results['predictions'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                   xticklabels=['Not Planet', 'Planet'],
                   yticklabels=['Not Planet', 'Planet'])

        ax.set_title('ENSEMBLE', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

        plt.tight_layout()
        output_path = self.output_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_roc_curves(self, predictor: EnsemblePredictor, X_test, y_test):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get predictions
        results = predictor.predict_ensemble(X_test)

        # Plot individual models
        for model_name, proba in results['individual_probabilities'].items():
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC={roc_auc:.3f})',
                   linewidth=2, alpha=0.7)

        # Plot ensemble
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'ENSEMBLE (AUC={roc_auc:.3f})',
               linewidth=3, color='black', linestyle='--')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - All Models (Test Set)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'roc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_feature_importance(self, ensemble_importance):
        """Plot ensemble feature importance from SHAP."""
        fig, ax = plt.subplots(figsize=(10, 12))

        # Get top 20 features
        top_features = ensemble_importance[:20]
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]

        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

        ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value| (average impact on model output)', fontsize=11)
        ax.set_title('Top 20 Features - Ensemble Importance', fontsize=14, fontweight='bold')

        # Add value labels
        for i, v in enumerate(importances):
            ax.text(v, i, f' {v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        output_path = self.output_dir / 'feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_shap_summary(self, predictor: EnsemblePredictor, X_sample, model_name='lightgbm'):
        """Create SHAP summary plot for a model."""
        if model_name not in predictor.explainers:
            print(f"No explainer for {model_name}, skipping SHAP summary...")
            return

        explainer = predictor.explainers[model_name]

        # Calculate SHAP values
        if isinstance(predictor.models[model_name], lgb.Booster):
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        elif isinstance(predictor.models[model_name], xgb.Booster):
            dmatrix = xgb.DMatrix(X_sample)
            shap_values = explainer.shap_values(dmatrix)
        else:
            shap_values = explainer.shap_values(X_sample)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]

        # Create summary plot
        plt.figure(figsize=(10, 12))
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = self.output_dir / f'shap_summary_{model_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

    def plot_confidence_distribution(self, predictor: EnsemblePredictor, X_test, y_test):
        """Plot distribution of confidence scores."""
        results = predictor.predict_ensemble(X_test)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Confidence distribution
        ax = axes[0]
        ax.hist(results['confidence'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Ensemble Confidence Scores', fontsize=14, fontweight='bold')
        ax.axvline(results['confidence'].mean(), color='red', linestyle='--',
                  label=f'Mean={results["confidence"].mean():.3f}')
        ax.legend()

        # Confidence vs Correctness
        ax = axes[1]
        correct = (results['predictions'] == y_test).astype(int)

        ax.scatter(results['confidence'][correct == 1], results['probabilities'][correct == 1],
                  c='green', alpha=0.5, label='Correct', s=20)
        ax.scatter(results['confidence'][correct == 0], results['probabilities'][correct == 0],
                  c='red', alpha=0.5, label='Incorrect', s=20)

        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Prediction Probability', fontsize=12)
        ax.set_title('Confidence vs Prediction Quality', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / 'confidence_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()


def main():
    """Create all visualizations."""
    print("="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv("data/processed/features_test.csv")

    exclude_cols = ['target', 'disposition', 'kepid', 'koi_name']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[feature_cols].copy()
    y_test = test_df['target'].copy()

    # Clean data
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        if X_test[col].isnull().any():
            X_test[col].fillna(X_test[col].median(), inplace=True)

    # Load training data for SHAP
    print("Loading training data...")
    train_df = pd.read_csv("data/processed/features_train.csv")
    X_train = train_df[feature_cols].copy()
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        if X_train[col].isnull().any():
            X_train[col].fillna(X_train[col].median(), inplace=True)

    # Initialize suite
    viz = VisualizationSuite()

    # Create predictor
    predictor = EnsemblePredictor()
    predictor.load_models()
    predictor.create_shap_explainers(X_train, sample_size=100)

    print("\n1. Creating model comparison plot...")
    viz.plot_model_comparison()

    print("\n2. Creating confusion matrices...")
    viz.plot_confusion_matrices(predictor, X_test, y_test)

    print("\n3. Creating ROC curves...")
    viz.plot_roc_curves(predictor, X_test, y_test)

    print("\n4. Creating confidence analysis...")
    viz.plot_confidence_distribution(predictor, X_test, y_test)

    print("\n5. Getting ensemble feature importance...")
    X_sample = X_test.sample(n=100, random_state=42)
    results = predictor.predict_and_explain(X_sample, X_train, create_explainers=False)

    print("\n6. Creating feature importance plot...")
    viz.plot_feature_importance(results['ensemble_feature_importance'])

    print("\n7. Creating SHAP summary plots...")
    viz.plot_shap_summary(predictor, X_sample, model_name='lightgbm')
    viz.plot_shap_summary(predictor, X_sample, model_name='xgboost')
    viz.plot_shap_summary(predictor, X_sample, model_name='random_forest')

    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nVisualizations saved to: {viz.output_dir}")
    print("\nGenerated plots:")
    for file in sorted(viz.output_dir.glob("*.png")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()
