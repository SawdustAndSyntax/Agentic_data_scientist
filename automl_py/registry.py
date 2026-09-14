from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def default_models(task):
    return (
        ["logistic", "rf", "extra_trees", "hist_gb", "svm_rbf", "knn", "mlp", "tree"]
        if task == "classification"
        else ["ridge", "lasso", "elastic_net", "rf", "extra_trees", "hist_gb", "svm_rbf", "knn", "mlp", "tree"]
    )


def build_estimator(name, task, random_state):
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as e:
            raise ImportError("Install automl-py[boost] for xgboost") from e
        return (XGBClassifier if task == "classification" else XGBRegressor)(
            n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=1
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as e:
            raise ImportError("Install automl-py[boost] for lightgbm") from e
        return (LGBMClassifier if task == "classification" else LGBMRegressor)(
            n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=random_state, n_jobs=1, verbose=-1
        )
    if name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as e:
            raise ImportError("Install automl-py[boost] for catboost") from e
        return (CatBoostClassifier if task == "classification" else CatBoostRegressor)(
            iterations=500, learning_rate=0.05, depth=6, random_seed=random_state, verbose=False, thread_count=1
        )
    if task == "classification":
        m = {
            "logistic": LogisticRegression(max_iter=3000),
            "rf": RandomForestClassifier(n_estimators=400, random_state=random_state, n_jobs=1),
            "extra_trees": ExtraTreesClassifier(n_estimators=400, random_state=random_state, n_jobs=1),
            "hist_gb": HistGradientBoostingClassifier(random_state=random_state),
            "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
            "knn": KNeighborsClassifier(),
            "mlp": MLPClassifier(max_iter=1500, random_state=random_state),
            "tree": DecisionTreeClassifier(random_state=random_state),
        }
    else:
        m = {
            "ridge": Ridge(),
            "lasso": Lasso(max_iter=10000),
            "elastic_net": ElasticNet(max_iter=10000),
            "rf": RandomForestRegressor(n_estimators=400, random_state=random_state, n_jobs=1),
            "extra_trees": ExtraTreesRegressor(n_estimators=400, random_state=random_state, n_jobs=1),
            "hist_gb": HistGradientBoostingRegressor(random_state=random_state),
            "svm_rbf": SVR(kernel="rbf"),
            "knn": KNeighborsRegressor(),
            "mlp": MLPRegressor(max_iter=1500, random_state=random_state),
            "tree": DecisionTreeRegressor(random_state=random_state),
        }
    if name not in m:
        raise KeyError(name)
    return m[name]


def parameter_space(name, task):
    common = {
        "rf": {"model__max_depth": [None, 6, 12, 20], "model__min_samples_leaf": [1, 2, 5]},
        "extra_trees": {"model__max_depth": [None, 8, 16], "model__min_samples_leaf": [1, 2, 5]},
        "hist_gb": {"model__learning_rate": [0.03, 0.08, 0.15], "model__max_leaf_nodes": [15, 31, 63]},
        "svm_rbf": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto"]},
        "knn": {"model__n_neighbors": [3, 5, 9, 15, 25], "model__weights": ["uniform", "distance"]},
        "tree": {"model__max_depth": [None, 4, 8, 16], "model__min_samples_leaf": [1, 2, 5, 10]},
        "xgboost": {"model__max_depth": [3, 5, 8], "model__learning_rate": [0.02, 0.05, 0.1]},
        "lightgbm": {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.02, 0.05, 0.1]},
        "catboost": {"model__depth": [4, 6, 8], "model__learning_rate": [0.02, 0.05, 0.1]},
    }
    if task == "classification":
        common["logistic"] = {"model__C": [0.01, 0.1, 1, 10, 100]}
    else:
        common["ridge"] = {"model__alpha": [0.01, 0.1, 1, 10, 100]}
        common["lasso"] = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1]}
        common["elastic_net"] = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1], "model__l1_ratio": [0.1, 0.5, 0.9]}
    return common.get(name, {})
