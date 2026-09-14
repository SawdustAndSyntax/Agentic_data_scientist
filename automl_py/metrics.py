import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def scorer_name(metric: str) -> str:
    return {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1_weighted",
    }[metric]


def higher_is_better(metric: str) -> bool:
    return metric not in {"rmse", "mae"}


def evaluate(metric, y_true, y_pred, y_prob=None) -> float:
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, average="weighted"))
    if metric == "roc_auc":
        if y_prob is None:
            raise ValueError("roc_auc requires scores")
        if getattr(y_prob, "ndim", 1) == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr" if getattr(y_prob, "ndim", 1) == 2 else "raise"))
    raise KeyError(metric)
