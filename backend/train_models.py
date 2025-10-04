"""
Train ensemble of 5 ML models for exoplanet detection.
Models: LightGBM, XGBoost, Random Forest, AdaBoost, Extra Trees
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb


class EnsembleTrainer:
    """Train and evaluate ensemble of 5 diverse models."""

    def __init__(self, data_dir: str = "data/processed", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models = {}
        self.metrics = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load feature matrices."""
        print("Loading feature matrices...")

        train_df = pd.read_csv(self.data_dir / "features_train.csv")
        val_df = pd.read_csv(self.data_dir / "features_val.csv")
        test_df = pd.read_csv(self.data_dir / "features_test.csv")

        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")

        return train_df, val_df, test_df

    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple:
        """
        Prepare X, y splits and handle missing values.

        Returns:
            (X_train, y_train, X_val, y_val, X_test, y_test, feature_names)
        """
        print("\nPreparing features...")

        # Columns to exclude from features
        exclude_cols = ['target', 'disposition', 'kepid', 'koi_name']

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        # Extract X and y
        X_train = train_df[feature_cols].copy()
        y_train = train_df['target'].copy()

        X_val = val_df[feature_cols].copy()
        y_val = val_df['target'].copy()

        X_test = test_df[feature_cols].copy()
        y_test = test_df['target'].copy()

        # Replace inf values with NaN first
        print(f"Replacing inf values...")
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Fill missing values with median (simple imputation)
        print(f"Filling missing values...")
        for col in feature_cols:
            if X_train[col].isnull().any():
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_val[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)

        print(f"Features: {len(feature_cols)}")
        print(f"Class distribution (train): {y_train.value_counts().to_dict()}")

        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> lgb.Booster:
        """Train LightGBM model."""
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM")
        print("="*60)

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Parameters
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])  # Handle class imbalance
        }

        # Train
        print("Training...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )

        print(f"Best iteration: {model.best_iteration}")

        return model

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> xgb.Booster:
        """Train XGBoost model."""
        print("\n" + "="*60)
        print("TRAINING XGBOOST")
        print("="*60)

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }

        # Train
        print("Training...")
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )

        print(f"Best iteration: {model.best_iteration}")

        return model

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> RandomForestClassifier:
        """Train Random Forest model."""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST")
        print("="*60)

        # Calculate class weights
        class_weight = {
            0: len(y_train) / (2 * len(y_train[y_train == 0])),
            1: len(y_train) / (2 * len(y_train[y_train == 1]))
        }

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("Training...")
        model.fit(X_train, y_train)

        # Validation score
        val_score = model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

        return model

    def train_adaboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> AdaBoostClassifier:
        """Train AdaBoost model."""
        print("\n" + "="*60)
        print("TRAINING ADABOOST")
        print("="*60)

        base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)

        model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42,
            algorithm='SAMME'
        )

        print("Training...")
        model.fit(X_train, y_train)

        # Validation score
        val_score = model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

        return model

    def train_extra_trees(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> ExtraTreesClassifier:
        """Train Extra Trees model."""
        print("\n" + "="*60)
        print("TRAINING EXTRA TREES")
        print("="*60)

        # Calculate class weights
        class_weight = {
            0: len(y_train) / (2 * len(y_train[y_train == 0])),
            1: len(y_train) / (2 * len(y_train[y_train == 1]))
        }

        model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("Training...")
        model.fit(X_train, y_train)

        # Validation score
        val_score = model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")

        return model

    def evaluate_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        dataset_name: str = "test"
    ) -> Dict:
        """Evaluate a single model."""

        # Get predictions
        if isinstance(model, lgb.Booster):
            y_pred_proba = model.predict(X, num_iteration=model.best_iteration)
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(X)
            y_pred_proba = model.predict(dmatrix, iteration_range=(0, model.best_iteration))
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'model': model_name,
            'dataset': dataset_name,
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred)),
            'f1': float(f1_score(y, y_pred)),
            'roc_auc': float(roc_auc_score(y, y_pred_proba))
        }

        return metrics

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all 5 models."""
        print("\n" + "="*60)
        print("TRAINING ENSEMBLE OF 5 MODELS")
        print("="*60)

        # Train LightGBM
        self.models['lightgbm'] = self.train_lightgbm(X_train, y_train, X_val, y_val)

        # Train XGBoost
        self.models['xgboost'] = self.train_xgboost(X_train, y_train, X_val, y_val)

        # Train Random Forest
        self.models['random_forest'] = self.train_random_forest(X_train, y_train, X_val, y_val)

        # Train AdaBoost
        self.models['adaboost'] = self.train_adaboost(X_train, y_train, X_val, y_val)

        # Train Extra Trees
        self.models['extra_trees'] = self.train_extra_trees(X_train, y_train, X_val, y_val)

        print("\n✓ All models trained successfully!")

    def evaluate_all_models(self, X_val, y_val, X_test, y_test):
        """Evaluate all models on validation and test sets."""
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)

        results = []

        for model_name, model in self.models.items():
            print(f"\n--- {model_name.upper()} ---")

            # Validation metrics
            val_metrics = self.evaluate_model(model, X_val, y_val, model_name, "validation")
            print(f"Validation - Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")

            # Test metrics
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name, "test")
            print(f"Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

            results.append(val_metrics)
            results.append(test_metrics)

            self.metrics[model_name] = {
                'validation': val_metrics,
                'test': test_metrics
            }

        return results

    def save_models(self, feature_names: List[str]):
        """Save all trained models."""
        print("\n--- Saving Models ---")

        for model_name, model in self.models.items():
            model_path = self.models_dir / f"{model_name}.pkl"

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            print(f"Saved: {model_path}")

        # Save feature names
        feature_path = self.models_dir / "feature_names.json"
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"Saved: {feature_path}")

        # Save metrics
        metrics_path = self.models_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Saved: {metrics_path}")


def main():
    """Run complete training pipeline."""

    trainer = EnsembleTrainer()

    # Load data
    train_df, val_df, test_df = trainer.load_data()

    # Prepare features
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = trainer.prepare_features(
        train_df, val_df, test_df
    )

    # Train all models
    trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Evaluate all models
    results = trainer.evaluate_all_models(X_val, y_val, X_test, y_test)

    # Save models
    trainer.save_models(feature_names)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nAll 5 models trained and saved to: {trainer.models_dir}")
    print("\nReady for Phase 4: Ensemble Integration & SHAP")


if __name__ == "__main__":
    main()
