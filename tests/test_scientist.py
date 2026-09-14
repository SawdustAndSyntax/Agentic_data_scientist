import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from automl_py import AutoMLScientist,AutoMLConfig,FeatureDiscovery,LeakageDetector

def test_feature_discovery_sales():
    d=FeatureDiscovery().opportunities(['date','store','price','sales'],'sales','weekly store ice cream sales')
    weather=d[d.concept=='weather'].iloc[0]
    assert not bool(weather.present)
    assert weather.priority=='high'

def test_leakage_detector_numeric():
    df=pd.DataFrame({'x':np.arange(50),'actual_result':np.arange(50),'target':np.arange(50)*2})
    out=LeakageDetector().detect(df,'target')
    assert (out.risk=='high').any()

def test_scientist_smoke(tmp_path):
    d=load_diabetes(as_frame=True); frame=d.frame.copy(); frame.loc[frame.index[:20],frame.columns[0]]=np.nan
    cfg=AutoMLConfig(task='regression',metric='r2',preprocessors=('original',),imputation_strategies=('median','mean'),feature_selection=('none',),models=('ridge',),cv_folds=3,tune=False,n_jobs=1,save_models=False,run_adversarial_validation=False,artifact_dir=str(tmp_path))
    result=AutoMLScientist(cfg,'predict disease progression').fit(frame,'target')
    assert len(result.automl.results)==2
    assert len(result.automl.best_results)==1
    assert not result.data_profile.empty
    assert not result.missingness.empty
    assert result.recommendations
