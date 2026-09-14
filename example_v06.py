from sklearn.datasets import load_diabetes
from automl_py import AutoMLConfig, AutonomousAutoMLScientist, FeatureAvailabilityRegistry

frame = load_diabetes(as_frame=True).frame

config = AutoMLConfig(
    task='regression', metric='rmse',
    preprocessors=('original','scale'),
    imputation_strategies=('median','mean'),
    feature_selection=('none',),
    models=('ridge','rf','extra_trees','hist_gb'),
    tune=True, n_jobs=-1,
    stability_repeats=6,
)

availability = FeatureAvailabilityRegistry({
    # Example: a post-outcome feature would be declared '+1d' and blocked.
    # 'actual_future_measurement': '+1d',
})

scientist = AutonomousAutoMLScientist(
    config,
    context='predict disease progression from available measurements',
    feature_availability=availability,
    feature_families={
        'body_composition': ['bmi'],
        'blood_pressure': ['bp'],
        'serum_markers': ['s1','s2','s3','s4','s5','s6'],
    },
)

result = scientist.fit(frame, 'target')
print(result.summary())
print(result.next_experiments())
