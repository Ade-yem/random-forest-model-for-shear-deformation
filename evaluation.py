import numpy as np # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore
from sklearn.model_selection import KFold, cross_val_score # type: ignore
from typing import Dict, Tuple, Any

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict[str, Any]:
    """
    Compute regression metrics including MAE, RMSE, R², and MAPE.
    
    Parameters
    ----------
    y_true : np.ndarray
        Array of actual ground truth values.
    y_pred : np.ndarray
        Array of predicted values by the model.
    label : str, optional
        Label indicating the model or dataset evaluated.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing computed valuation metrics.
    """
    mae: float  = float(mean_absolute_error(y_true, y_pred))
    rmse: float = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2: float   = float(r2_score(y_true, y_pred))
    mape: float = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0)
    
    if label:
        print(f"\n  [{label}]")
        print(f"    MAE   = {mae:.4f} mm")
        print(f"    RMSE  = {rmse:.4f} mm")
        print(f"    R²    = {r2:.4f}")
        print(f"    MAPE  = {mape:.2f} %")
        
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def run_cross_validation(model: Any, X: np.ndarray, y: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run k-fold cross validation on an estimator to validate performance
    robustness on smaller datasets (Section 3.3.4 of thesis).
    
    Parameters
    ----------
    model : Any
        An sklearn model instance (like RandomForestRegressor).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    k : int, optional
        Number of folds to run. Default is 5.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Arrays of R² scores and RMSE scores for each fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores: np.ndarray   = cross_val_score(model, X, y, cv=kf, scoring="r2", n_jobs=-1)
    rmse_scores: np.ndarray = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                          scoring="neg_mean_squared_error", n_jobs=-1))
                          
    print(f"\n  {k}-Fold Cross-Validation:")
    print(f"    R²   : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"    RMSE : {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f} mm")
    return r2_scores, rmse_scores
