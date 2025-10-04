"""
Ensemble prediction system with SHAP explainability.
Combines predictions from all 5 models with confidence scores and interpretability.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import shap


class EnsemblePredictor:
    """Ensemble prediction with SHAP explanations."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.feature_names = []
        self.explainers = {}

    def load_models(self):
        """Load all trained models."""
        print("Loading models...")

        model_names = ['lightgbm', 'xgboost', 'random_forest', 'adaboost', 'extra_trees']

        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}.pkl"

            if not model_path.exists():
                print(f"Warning: {model_name} not found, skipping...")
                continue

            with open(model_path, 'rb') as f:
                self.models[model_name] = pickle.load(f)

            print(f"✓ Loaded {model_name}")

        # Load feature names
        feature_path = self.models_dir / "feature_names.json"
        with open(feature_path, 'r') as f:
            self.feature_names = json.load(f)

        print(f"\n✓ Loaded {len(self.models)} models with {len(self.feature_names)} features")

    def predict_single_model(
        self,
        model,
        model_name: str,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions from a single model.

        Returns:
            (predictions, probabilities)
        """
        if isinstance(model, lgb.Booster):
            proba = model.predict(X, num_iteration=model.best_iteration)
            pred = (proba > 0.5).astype(int)
        elif isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X)
            proba = model.predict(dmatrix, iteration_range=(0, model.best_iteration))
            pred = (proba > 0.5).astype(int)
        else:
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1]

        return pred, proba

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        voting: str = 'soft'
    ) -> Dict:
        """
        Get ensemble predictions from all models.

        Args:
            X: Feature matrix
            voting: 'soft' (average probabilities) or 'hard' (majority vote)

        Returns:
            Dictionary with predictions, probabilities, and individual model outputs
        """
        if len(self.models) == 0:
            raise ValueError("No models loaded. Call load_models() first.")

        # Get predictions from each model
        individual_predictions = {}
        individual_probabilities = {}

        for model_name, model in self.models.items():
            pred, proba = self.predict_single_model(model, model_name, X)
            individual_predictions[model_name] = pred
            individual_probabilities[model_name] = proba

        # Ensemble voting
        if voting == 'soft':
            # Average probabilities
            all_probs = np.array(list(individual_probabilities.values()))
            ensemble_proba = np.mean(all_probs, axis=0)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)

        else:  # hard voting
            # Majority vote
            all_preds = np.array(list(individual_predictions.values()))
            ensemble_pred = (np.mean(all_preds, axis=0) > 0.5).astype(int)

            # For probability, still use average
            all_probs = np.array(list(individual_probabilities.values()))
            ensemble_proba = np.mean(all_probs, axis=0)

        # Calculate confidence metrics
        confidence = self._calculate_confidence(individual_probabilities)

        return {
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba,
            'confidence': confidence,
            'individual_predictions': individual_predictions,
            'individual_probabilities': individual_probabilities,
            'voting_method': voting
        }

    def _calculate_confidence(self, individual_probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate confidence score based on model agreement.

        High confidence = models agree strongly
        Low confidence = models disagree

        Returns:
            Confidence scores [0, 1] for each sample
        """
        all_probs = np.array(list(individual_probabilities.values()))

        # Calculate standard deviation across models
        std = np.std(all_probs, axis=0)

        # Low std = high agreement = high confidence
        # Map std [0, 0.5] to confidence [1, 0]
        confidence = 1 - (std / 0.5)
        confidence = np.clip(confidence, 0, 1)

        return confidence

    def create_shap_explainers(self, X_background: pd.DataFrame, sample_size: int = 100):
        """
        Create SHAP explainers for each model.

        Args:
            X_background: Background dataset for SHAP (training data sample)
            sample_size: Number of background samples to use
        """
        print("\nCreating SHAP explainers...")

        # Sample background data
        if len(X_background) > sample_size:
            background = X_background.sample(n=sample_size, random_state=42)
        else:
            background = X_background

        for model_name, model in self.models.items():
            print(f"Creating explainer for {model_name}...")

            try:
                if isinstance(model, lgb.Booster):
                    # TreeExplainer for LightGBM
                    self.explainers[model_name] = shap.TreeExplainer(model)

                elif isinstance(model, xgb.Booster):
                    # TreeExplainer for XGBoost
                    self.explainers[model_name] = shap.TreeExplainer(model)

                else:
                    # TreeExplainer for sklearn tree-based models
                    self.explainers[model_name] = shap.TreeExplainer(model)

                print(f"✓ Created explainer for {model_name}")

            except Exception as e:
                print(f"Warning: Could not create explainer for {model_name}: {e}")

        print(f"\n✓ Created {len(self.explainers)} SHAP explainers")

    def explain_predictions(
        self,
        X: pd.DataFrame,
        model_name: str = None
    ) -> Dict:
        """
        Get SHAP explanations for predictions.

        Args:
            X: Samples to explain
            model_name: Specific model to explain (if None, explain all)

        Returns:
            Dictionary with SHAP values and feature importance
        """
        if len(self.explainers) == 0:
            raise ValueError("No explainers created. Call create_shap_explainers() first.")

        explanations = {}

        # Explain specific model or all models
        models_to_explain = [model_name] if model_name else list(self.explainers.keys())

        for model_name in models_to_explain:
            if model_name not in self.explainers:
                continue

            explainer = self.explainers[model_name]

            # Calculate SHAP values
            if isinstance(self.models[model_name], (lgb.Booster, xgb.Booster)):
                # For boosted trees, handle differently
                if isinstance(self.models[model_name], lgb.Booster):
                    shap_values = explainer.shap_values(X)
                    # LightGBM returns list for binary classification, take positive class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                elif isinstance(self.models[model_name], xgb.Booster):
                    dmatrix = xgb.DMatrix(X)
                    shap_values = explainer.shap_values(dmatrix)
            else:
                # Sklearn models
                shap_values = explainer.shap_values(X)
                # For binary classification, take positive class
                if len(shap_values.shape) == 3:
                    shap_values = shap_values[:, :, 1]

            # Get feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)

            # Sort features by importance
            feature_ranking = sorted(
                zip(self.feature_names, feature_importance),
                key=lambda x: x[1],
                reverse=True
            )

            explanations[model_name] = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'feature_ranking': feature_ranking,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None
            }

        return explanations

    def get_ensemble_feature_importance(self, explanations: Dict) -> List[Tuple[str, float]]:
        """
        Aggregate feature importance across all models.

        Args:
            explanations: Output from explain_predictions()

        Returns:
            List of (feature_name, importance) sorted by importance
        """
        # Average importance across models
        all_importances = {}

        for model_name, exp_data in explanations.items():
            for feature, importance in exp_data['feature_ranking']:
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(importance)

        # Calculate mean importance
        ensemble_importance = {
            feature: np.mean(importances)
            for feature, importances in all_importances.items()
        }

        # Sort by importance
        ranking = sorted(
            ensemble_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranking

    def predict_and_explain(
        self,
        X: pd.DataFrame,
        X_background: pd.DataFrame = None,
        create_explainers: bool = False
    ) -> Dict:
        """
        Complete prediction and explanation pipeline.

        Args:
            X: Samples to predict
            X_background: Background data for SHAP (if creating explainers)
            create_explainers: Whether to create new explainers

        Returns:
            Complete results with predictions and explanations
        """
        # Get ensemble predictions
        predictions = self.predict_ensemble(X)

        # Create or use existing explainers
        if create_explainers and X_background is not None:
            self.create_shap_explainers(X_background)

        # Get explanations
        if len(self.explainers) > 0:
            explanations = self.explain_predictions(X)
            ensemble_importance = self.get_ensemble_feature_importance(explanations)
        else:
            explanations = {}
            ensemble_importance = []

        return {
            'predictions': predictions,
            'explanations': explanations,
            'ensemble_feature_importance': ensemble_importance
        }


def main():
    """Test ensemble predictor on test set."""

    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/features_test.csv")

    # Prepare features
    exclude_cols = ['target', 'disposition', 'kepid', 'koi_name']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols]

    X_test = test_df[feature_cols].copy()
    y_test = test_df['target'].copy()

    # Handle inf and missing values
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        if X_test[col].isnull().any():
            X_test[col].fillna(X_test[col].median(), inplace=True)

    # Load training data for background
    print("Loading training data for SHAP background...")
    train_df = pd.read_csv("data/processed/features_train.csv")
    X_train = train_df[feature_cols].copy()
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_cols:
        if X_train[col].isnull().any():
            X_train[col].fillna(X_train[col].median(), inplace=True)

    # Create predictor
    predictor = EnsemblePredictor()
    predictor.load_models()

    # Test on first 10 samples
    print("\n" + "="*60)
    print("TESTING ENSEMBLE PREDICTIONS")
    print("="*60)

    X_sample = X_test.head(10)
    y_sample = y_test.head(10)

    results = predictor.predict_and_explain(
        X_sample,
        X_background=X_train,
        create_explainers=True
    )

    print("\n--- Ensemble Predictions ---")
    pred_results = results['predictions']

    for i in range(len(X_sample)):
        true_label = "PLANET" if y_sample.iloc[i] == 1 else "NOT PLANET"
        pred_label = "PLANET" if pred_results['predictions'][i] == 1 else "NOT PLANET"
        confidence = pred_results['confidence'][i]
        probability = pred_results['probabilities'][i]

        print(f"\nSample {i+1}:")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label} (prob={probability:.3f}, confidence={confidence:.3f})")
        print(f"  Individual models:")
        for model_name, proba in pred_results['individual_probabilities'].items():
            print(f"    {model_name}: {proba[i]:.3f}")

    # Show top features
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES (Ensemble)")
    print("="*60)

    ensemble_importance = results['ensemble_feature_importance']
    for rank, (feature, importance) in enumerate(ensemble_importance[:10], 1):
        print(f"{rank:2d}. {feature:30s} {importance:.4f}")

    print("\n✓ Ensemble prediction system working!")
    print("Ready to integrate into API and create visualizations")


if __name__ == "__main__":
    main()
