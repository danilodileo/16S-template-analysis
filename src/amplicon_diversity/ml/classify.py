"""Machine learning on compositional microbiome data: CLR transform, CV, feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def clr_transform(table: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Centered log-ratio (CLR) transform of a compositional (samples x features) table.

    CLR(x)_i = log(x_i) - mean_j(log(x_j)), computed per sample. This is the
    standard way to make compositional abundance data usable by models that
    assume unconstrained, roughly-Euclidean features (e.g. random forests,
    logistic regression, PCA).
    """
    rel = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    shifted = rel + pseudocount
    log_vals = np.log(shifted)
    return log_vals.sub(log_vals.mean(axis=1), axis=0)


def cross_validate_classifier(
    table: pd.DataFrame,
    labels: pd.Series,
    model: ClassifierMixin | None = None,
    n_splits: int = 5,
    seed: int | None = 0,
) -> dict:
    """Stratified k-fold cross-validation of a classifier on CLR-transformed abundance data.

    Parameters
    ----------
    table : DataFrame
        Samples x features count/abundance table.
    labels : Series
        Binary or multiclass sample labels, index-aligned with ``table``.
    model : sklearn classifier, optional
        Defaults to a ``RandomForestClassifier(n_estimators=500)``.
    n_splits : int
        Number of CV folds.

    Returns
    -------
    dict with keys:
        "accuracy": mean out-of-fold accuracy
        "roc_auc": out-of-fold ROC AUC (binary labels only, else None)
        "predictions": Series of out-of-fold predicted labels
        "probabilities": DataFrame of out-of-fold predicted probabilities per class
        "fitted_model": the model refit on the full dataset
    """
    labels = labels.reindex(table.index)
    if model is None:
        model = RandomForestClassifier(n_estimators=500, random_state=seed)

    x = clr_transform(table).values
    y = labels.values
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_pred = cross_val_predict(clone(model), x, y, cv=cv, method="predict")
    classes = np.unique(y)
    roc_auc = None
    proba_df = None
    try:
        oof_proba = cross_val_predict(clone(model), x, y, cv=cv, method="predict_proba")
        proba_df = pd.DataFrame(oof_proba, index=table.index, columns=classes)
        if len(classes) == 2:
            roc_auc = float(roc_auc_score(y, oof_proba[:, 1]))
    except AttributeError:
        pass

    fitted_model = clone(model).fit(x, y)

    return {
        "accuracy": float(accuracy_score(y, oof_pred)),
        "roc_auc": roc_auc,
        "predictions": pd.Series(oof_pred, index=table.index, name="predicted"),
        "probabilities": proba_df,
        "fitted_model": fitted_model,
    }


def feature_importance(fitted_model, feature_names: list[str]) -> pd.Series:
    """Extract a sorted feature-importance Series from a fitted tree-based or linear model."""
    if hasattr(fitted_model, "feature_importances_"):
        values = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        coef = fitted_model.coef_
        values = np.abs(coef[0]) if coef.ndim == 2 else np.abs(coef)
    else:
        raise ValueError("model has neither feature_importances_ nor coef_")
    return pd.Series(values, index=feature_names, name="importance").sort_values(ascending=False)
