import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score


def print_metrics(y_true, y_pred, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(model_name)
    print(f"Sensitivity: {tp / (tp + fn) * 100:.1f}%")
    print(f"Specificity: {tn / (tn + fp) * 100:.1f}%")
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.1f}%")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_pred) * 100:.1f}%")
    print()


def cross_validate(estimator, X: pd.DataFrame, y: pd.Series, neg: int, k=10):
    metrics = {
        "Specificity": make_scorer(recall_score, pos_label=1),
        "Sensitivity": make_scorer(recall_score, pos_label=neg),
        "Accuracy": make_scorer(accuracy_score),
        "ROC-AUC": make_scorer(roc_auc_score),
    }
    for name, func in metrics.items():
        scores: np.ndarray = cross_val_score(
            estimator=estimator, X=X, y=y, cv=k, scoring=func
        )
        mu = scores.mean()
        sigma = scores.std()
        print(f"{name}: {mu * 100:.1f}% +- {sigma * 100:.1f}%")
