from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa:F401
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


def _numeric_imputer(strategy, add_indicator):
    if strategy in {"median", "mean", "most_frequent"}:
        return SimpleImputer(strategy=strategy, add_indicator=add_indicator)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5, add_indicator=add_indicator)
    if strategy == "iterative":
        return IterativeImputer(random_state=100, max_iter=10)
    raise ValueError(strategy)


def build_preprocessor(
    X, mode, pca_variance=0.95, interaction_terms=False, interaction_degree=2, imputation_strategy="median", add_missing_indicators=True
):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num = [("impute", _numeric_imputer(imputation_strategy, add_missing_indicators))]
    if mode in {"scale", "pca"}:
        num.append(("scale", StandardScaler()))
    if interaction_terms:
        num.append(("interactions", PolynomialFeatures(degree=interaction_degree, interaction_only=True, include_bias=False)))
    cat = [
        ("impute", SimpleImputer(strategy="most_frequent", add_indicator=add_missing_indicators)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    tr = ColumnTransformer(
        [("num", Pipeline(num), numeric), ("cat", Pipeline(cat), categorical)], remainder="drop", verbose_feature_names_out=False
    )
    if mode == "pca":
        return Pipeline([("columns", tr), ("pca_scale", StandardScaler()), ("pca", PCA(n_components=pca_variance, svd_solver="full"))])
    return tr


def build_feature_selector(task, method, k="all"):
    if method == "none":
        return "passthrough"
    if method not in {"f_test", "mutual_info"}:
        raise ValueError(method)
    fn = (
        (f_classif if method == "f_test" else mutual_info_classif)
        if task == "classification"
        else (f_regression if method == "f_test" else mutual_info_regression)
    )
    return SelectKBest(score_func=fn, k=k)
