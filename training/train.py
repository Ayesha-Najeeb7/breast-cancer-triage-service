import argparse,json,os,subprocess
from pathlib import Path
import joblib,pandas as pd,yaml,sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,recall_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold,cross_validate,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def run(data_path,out,version,params_path='params.yaml'):
 p=yaml.safe_load(Path(params_path).read_text(encoding='utf-8')); seed=int(p['seed']); hp=p['train']; df=pd.read_csv(data_path); features=[c for c in df.columns if c!='target']
 if len(features)!=30 or df.isnull().values.any(): raise ValueError('CSV must have 30 features and no nulls')
 X=df[features]; y=(df.target==0).astype(int); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=seed,stratify=y)
 models={'logistic_regression':Pipeline([('imputer',SimpleImputer()),('scale',StandardScaler()),('model',LogisticRegression(C=hp['logistic_c'],max_iter=2000,random_state=seed))]),'random_forest':Pipeline([('imputer',SimpleImputer()),('model',RandomForestClassifier(n_estimators=hp['forest_estimators'],max_depth=hp['forest_max_depth'],class_weight='balanced',random_state=seed,n_jobs=1))])}
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); rows=[]
 for name,m in models.items():
  v=cross_validate(m,Xtr,ytr,cv=cv,scoring={'recall':'recall','accuracy':'accuracy','roc_auc':'roc_auc'},n_jobs=1); fit=clone(m).fit(Xtr,ytr); prob=fit.predict_proba(Xte)[:,1]; pred=(prob>=float(p['decision_threshold'])).astype(int)
  rows.append({'model_family':name,'cv_malignant_recall_mean':float(v['test_recall'].mean()),'cv_accuracy_mean':float(v['test_accuracy'].mean()),'cv_roc_auc_mean':float(v['test_roc_auc'].mean()),'test_malignant_recall':float(recall_score(yte,pred)),'test_accuracy':float(accuracy_score(yte,pred)),'test_roc_auc':float(roc_auc_score(yte,prob)),'pipeline':fit})
 best=max(rows,key=lambda r:r['cv_malignant_recall_mean']); out=Path(out); out.mkdir(parents=True,exist_ok=True); metrics=[{k:v for k,v in r.items() if k!='pipeline'} for r in rows]
 try: sha=subprocess.check_output(['git','rev-parse','--short','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
 except Exception: sha=os.getenv('GIT_SHA','unknown')
 card={'model_version':version,'model_family':best['model_family'],'feature_names':features,'metrics':metrics,'selected_test_metrics':{k:best[k] for k in ('test_malignant_recall','test_accuracy','test_roc_auc')},'seed':seed,'decision_threshold':float(p['decision_threshold']),'feature_count':30,'sklearn_version':sklearn.__version__,'git_commit':sha}
 joblib.dump(best['pipeline'],out/'model.joblib'); (out/'model_card.json').write_text(json.dumps(card,indent=2),encoding='utf-8'); (out/'metrics.json').write_text(json.dumps({'models':metrics,'selected_model':best['model_family']},indent=2),encoding='utf-8'); print(json.dumps(card,indent=2))
 if best['test_malignant_recall']<.95: raise SystemExit('Selected test malignant recall below 0.95')
if __name__=='__main__':
 a=argparse.ArgumentParser(); a.add_argument('--data-path',default='data/raw/breast_cancer.csv'); a.add_argument('--output-dir',default='artifacts'); a.add_argument('--model-version',default='1.0.0'); x=a.parse_args(); run(x.data_path,x.output_dir,x.model_version)
