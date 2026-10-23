"""Export sklearn breast-cancer data."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

def export(path):
    data=load_breast_cancer(); frame=pd.DataFrame(data.data,columns=[n.replace(' ','_') for n in data.feature_names]); frame['target']=data.target
    if frame.shape[1]!=31 or frame.isnull().values.any(): raise ValueError('Expected 30 features plus target and no nulls')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(path,index=False); print(f'Exported {len(frame)} rows to {path}')
if __name__=='__main__': export('data/raw/breast_cancer.csv')
