import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, average_precision_score,
                            confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import os

def train_xgboost(X_train, y_train, use_class_weighting=False, random_state=42):
    """Train XGBoost model with hyperparameter tuning"""
    print(f"\nTraining XGBoost (Class weighting: {use_class_weighting})...")

    if use_class_weighting:
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0]
        print(f"Calculated scale_pos_weight: {scale_pos_weight:.3f}")
    else:
        scale_pos_weight = 1

    base_params = {
        'random_state': random_state,
        'eval_metric': 'logloss',
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': -1
    }

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.05, 0.1, 0.15, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = XGBClassifier(**base_params)
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring='average_precision',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    random_search.fit(X_train, y_train)
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validated PR-AUC: {random_search.best_score_:.4f}")
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        print(f"\n=== {model_name} Evaluation ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Confusion Matrix:\n{np.array(metrics['confusion_matrix'])}")
        return metrics

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def save_risk_scores(model, X_test, file_path):
    """Save risk scores for test set"""
    try:
        risk_scores = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)

        results_df = pd.DataFrame({
            'patient_id': X_test.index,
            'risk_score': risk_scores,
            'prediction': predictions,
            'risk_category': pd.cut(risk_scores,
                                  bins=[0, 0.3, 0.7, 1.0],
                                  labels=['Low', 'Moderate', 'High'])
        })

        results_df.to_csv(file_path, index=False)
        print(f"Risk scores saved to {file_path}")

    except Exception as e:
        print(f"Error saving risk scores: {e}")

def select_best_model(models_metrics):
    """Select best model based on comprehensive scoring"""
    best_model_info = None
    best_score = -1

    print("\n=== MODEL SELECTION ===")
    for model_info in models_metrics:
        if model_info['metrics'] is None:
            continue

        metrics = model_info['metrics']
        composite_score = (
            0.4 * metrics['pr_auc'] +
            0.3 * metrics['f1_score'] +
            0.2 * metrics['roc_auc'] +
            0.1 * metrics['recall']
        )

        print(f"{model_info['name']}: Composite Score = {composite_score:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1_score']:.4f}")

        if composite_score > best_score:
            best_score = composite_score
            best_model_info = model_info

    return best_model_info, best_score