from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor, build_feature_selector
from .registry import build_estimator

from .config import AutoMLConfig
from .core import AutoML
from .scientist import AutoMLScientist, ScientistResult
from .stability import model_stability
from .uncertainty import SplitConformalRegressor
from .experiments import feature_family_value
from .planner import NextExperimentPlanner
from .temporal import FeatureAvailabilityRegistry

@dataclass
class AutonomousScientistResult:
    scientist: ScientistResult
    experiment_plan: pd.DataFrame
    stability: dict | None
    uncertainty: dict | None
    temporal_audit: pd.DataFrame
    information_value: pd.DataFrame

    @property
    def automl(self): return self.scientist.automl
    def best_model(self,target): return self.scientist.automl.best_model(target)
    def next_experiments(self,n: int = 5): return self.experiment_plan.head(n)
    def summary(self):
        parts=[self.scientist.summary(),'','Autonomous experiment plan','='*26]
        for _,r in self.experiment_plan.head(8).iterrows():
            parts.append(f"P{int(r.priority)} [{r.category}] {r.experiment}: {r.rationale}")
        if self.uncertainty:
            parts += ['',f"Conformal interval: {(1-self.uncertainty['alpha'])*100:.0f}% nominal, radius={self.uncertainty['radius']:.4f}, empirical coverage={self.uncertainty['empirical_coverage']:.3f}"]
        return '\n'.join(parts)

class AutonomousAutoMLScientist:
    """v0.6 orchestration layer.

    It does not fabricate or automatically purchase external data. It identifies missing-signal
    hypotheses, proves value when candidate feature sets are supplied, and recommends the next
    controlled experiment based on validity, robustness, information value and residual error.
    """
    def __init__(self, config: AutoMLConfig | None = None, context: str = '',
                 feature_availability: FeatureAvailabilityRegistry | None = None,
                 feature_families: dict[str,list[str]] | None = None):
        self.config=config or AutoMLConfig(); self.context=context
        self.feature_availability=feature_availability or FeatureAvailabilityRegistry()
        self.feature_families=feature_families or {}
        self.result_=None

    def fit(self, data: pd.DataFrame, target_names: str | list[str]):
        c=self.config; df=pd.DataFrame(data).copy(); targets=[target_names] if isinstance(target_names,str) else list(target_names); target=targets[0]
        Xraw=df.drop(columns=targets)
        temporal_audit=self.feature_availability.audit(Xraw.columns)
        unavailable=temporal_audit.loc[temporal_audit.temporal_leakage_risk,'feature'].tolist()
        if unavailable:
            if c.temporal_strict:
                raise ValueError('Temporal leakage risk; unavailable at prediction time: '+', '.join(unavailable))
            df=df.drop(columns=unavailable)

        base=AutoMLScientist(c,self.context).fit(df,targets)
        bestrow=base.automl.best_results[base.automl.best_results.target==target]
        stability=None; uncertainty=None; info=pd.DataFrame()
        if not bestrow.empty:
            tag=bestrow.iloc[0].tag; model=base.automl.models[tag]
            X=df.drop(columns=targets); y=df[target]; mask=y.notna(); X=X.loc[mask]; y=y.loc[mask]
            if c.run_stability_analysis:
                try:
                    stability=model_stability(model,X,y,task=c.task,metric=c.resolved_metric(),repeats=c.stability_repeats,test_size=c.test_size,random_state=c.random_state,n_jobs=1)
                except Exception: stability=None
            if c.task=='regression' and c.run_uncertainty:
                try:
                    Xtr,Xte,ytr,yte=train_test_split(
                        X,y,test_size=c.test_size,random_state=c.random_state
                    )
                    conf=SplitConformalRegressor(
                        model,alpha=c.uncertainty_alpha,random_state=c.random_state
                    ).fit(Xtr,ytr)
                    uncertainty={
                        'alpha':c.uncertainty_alpha,
                        'radius':conf.radius_,
                        'empirical_coverage':conf.empirical_coverage(Xte,yte),
                        'evaluation':'held-out test split not used for conformal fitting/calibration',
                    }
                except Exception: uncertainty=None
            if self.feature_families:
                try:
                    row=bestrow.iloc[0]
                    params=json.loads(row.best_params) if row.best_params else {}
                    def factory(Xsubset):
                        pipe=Pipeline([
                            ('prep',build_preprocessor(
                                Xsubset,row.preprocess,c.pca_variance,c.interaction_terms,
                                c.interaction_degree,row.imputation,c.add_missing_indicators
                            )),
                            ('select',build_feature_selector(c.task,row.selection,c.feature_selection_k)),
                            ('model',build_estimator(row.model,c.task,c.random_state)),
                        ])
                        if params: pipe.set_params(**params)
                        return pipe
                    info=feature_family_value(
                        factory,X,y,cv=AutoML(c)._cv(),metric=c.resolved_metric(),
                        feature_families=self.feature_families,n_jobs=c.n_jobs
                    )
                except Exception: info=pd.DataFrame()

        plan=NextExperimentPlanner().plan(metric=c.resolved_metric(),leakage=base.leakage,missingness=base.missingness,opportunities=base.feature_opportunities,drift=base.drift,residuals=base.residual_diagnostics,stability=stability,temporal_audit=temporal_audit,information_value=info)
        result=AutonomousScientistResult(base,plan,stability,uncertainty,temporal_audit,info); self.result_=result
        out=Path(c.artifact_dir); out.mkdir(parents=True,exist_ok=True)
        plan.to_csv(out/'next_experiments.csv',index=False); temporal_audit.to_csv(out/'temporal_availability_audit.csv',index=False)
        if not info.empty: info.to_csv(out/'feature_family_value.csv',index=False)
        if stability:
            stability['scores'].to_csv(out/'stability_scores.csv',index=False)
            stability['feature_stability'].to_csv(out/'feature_stability.csv',index=False)
            (out/'stability_summary.json').write_text(json.dumps(stability['score_summary'],indent=2))
        if uncertainty: (out/'uncertainty_summary.json').write_text(json.dumps(uncertainty,indent=2))
        (out/'autonomous_summary.txt').write_text(result.summary())
        return result
