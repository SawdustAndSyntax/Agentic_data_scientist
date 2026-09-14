from sklearn.datasets import load_diabetes
from automl_py import AutoMLScientist, AutoMLConfig

df = load_diabetes(as_frame=True).frame

config = AutoMLConfig(
    task="regression",
    metric="rmse",
    preprocessors=("original", "scale"),
    imputation_strategies=("median", "mean", "knn"),
    feature_selection=("none",),
    models=("ridge", "rf", "extra_trees", "hist_gb"),
    tune=True,
    n_jobs=-1,
)

result = AutoMLScientist(
    config,
    context="predict continuous disease progression outcome",
).fit(df, "target")

print(result.summary())
