import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def adversarial_validation(train_df,test_df,random_state=100):
    common=[c for c in train_df.columns if c in test_df.columns]; a=train_df[common].copy(); b=test_df[common].copy(); X=pd.concat([a,b],ignore_index=True); y=np.r_[np.zeros(len(a),dtype=int),np.ones(len(b),dtype=int)]
    nums=X.select_dtypes(include=['number','bool']).columns.tolist(); cats=[c for c in X.columns if c not in nums]
    prep=ColumnTransformer([('num',SimpleImputer(strategy='median'),nums),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('enc',OneHotEncoder(handle_unknown='ignore'))]),cats)])
    model=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=200,max_depth=7,random_state=random_state,n_jobs=1))])
    Xtr,Xva,ytr,yva=train_test_split(X,y,test_size=.3,random_state=random_state,stratify=y); model.fit(Xtr,ytr); auc=float(roc_auc_score(yva,model.predict_proba(Xva)[:,1])); pi=permutation_importance(model,Xva,yva,scoring='roc_auc',n_repeats=3,random_state=random_state,n_jobs=1)
    imp=pd.DataFrame({'feature':X.columns,'importance':pi.importances_mean,'importance_std':pi.importances_std}).sort_values('importance',ascending=False)
    return {'auc':auc,'shift_severity':'low' if auc<.6 else 'moderate' if auc<.75 else 'high','feature_importance':imp,'model':model}
