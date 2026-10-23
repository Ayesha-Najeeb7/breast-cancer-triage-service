import json,logging,os
from pathlib import Path
from contextlib import asynccontextmanager
import joblib,pandas as pd
try: import redis
except ImportError: redis=None
from fastapi import FastAPI
from pydantic import BaseModel,ConfigDict,Field
logging.basicConfig(level=os.getenv('LOG_LEVEL','INFO'),format='%(asctime)s %(levelname)s %(message)s'); log=logging.getLogger('triage')
MODEL_PATH=Path(os.getenv('MODEL_PATH','artifacts/model.joblib')); CARD_PATH=MODEL_PATH.with_name('model_card.json'); THRESHOLD=float(os.getenv('DECISION_THRESHOLD','0.5')); model=None; card={}; cache=None
FEATURES='mean_radius mean_texture mean_perimeter mean_area mean_smoothness mean_compactness mean_concavity mean_concave_points mean_symmetry mean_fractal_dimension radius_error texture_error perimeter_error area_error smoothness_error compactness_error concavity_error concave_points_error symmetry_error fractal_dimension_error worst_radius worst_texture worst_perimeter worst_area worst_smoothness worst_compactness worst_concavity worst_concave_points worst_symmetry worst_fractal_dimension'.split()
class Record(BaseModel):
 model_config=ConfigDict(extra='forbid')
 mean_radius:float=Field(ge=0); mean_texture:float=Field(ge=0); mean_perimeter:float=Field(ge=0); mean_area:float=Field(ge=0); mean_smoothness:float=Field(ge=0); mean_compactness:float=Field(ge=0); mean_concavity:float=Field(ge=0); mean_concave_points:float=Field(ge=0); mean_symmetry:float=Field(ge=0); mean_fractal_dimension:float=Field(ge=0); radius_error:float=Field(ge=0); texture_error:float=Field(ge=0); perimeter_error:float=Field(ge=0); area_error:float=Field(ge=0); smoothness_error:float=Field(ge=0); compactness_error:float=Field(ge=0); concavity_error:float=Field(ge=0); concave_points_error:float=Field(ge=0); symmetry_error:float=Field(ge=0); fractal_dimension_error:float=Field(ge=0); worst_radius:float=Field(ge=0); worst_texture:float=Field(ge=0); worst_perimeter:float=Field(ge=0); worst_area:float=Field(ge=0); worst_smoothness:float=Field(ge=0); worst_compactness:float=Field(ge=0); worst_concavity:float=Field(ge=0); worst_concave_points:float=Field(ge=0); worst_symmetry:float=Field(ge=0); worst_fractal_dimension:float=Field(ge=0)
@asynccontextmanager
async def lifespan(app):
 global model,card,cache
 if not MODEL_PATH.exists(): log.error('Required model artifact missing: %s',MODEL_PATH); raise RuntimeError(f'Missing model artifact: {MODEL_PATH}')
 model=joblib.load(MODEL_PATH); card=json.loads(CARD_PATH.read_text(encoding='utf-8')) if CARD_PATH.exists() else {}
 if os.getenv('REDIS_HOST') and redis: cache=redis.Redis(host=os.getenv('REDIS_HOST'),port=int(os.getenv('REDIS_PORT','6379')),decode_responses=True,socket_connect_timeout=.5,socket_timeout=.5)
 yield
app=FastAPI(title='Breast Cancer Triage API',version=os.getenv('MODEL_VERSION','1.0.0'),lifespan=lifespan)
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict')
def predict(record:Record):
 global cache
 vals=record.model_dump(); malignant=float(model.predict_proba(pd.DataFrame([vals],columns=card.get('feature_names',FEATURES)))[0][1])
 if cache:
  try: cache.incr('predictions')
  except redis.RedisError: log.exception('Redis unavailable; prediction still served'); cache=None
 return {'label':'malignant' if malignant>=THRESHOLD else 'benign','malignant_probability':round(malignant,4),'threshold':THRESHOLD,'model_version':os.getenv('MODEL_VERSION',card.get('model_version','1.0.0'))}
@app.get('/model/info')
def info(): return {'model_version':os.getenv('MODEL_VERSION',card.get('model_version','1.0.0')),'model_family':card.get('model_family','unknown'),'scikit_learn_version':card.get('sklearn_version','unknown'),'git_commit':os.getenv('GIT_SHA',card.get('git_commit','unknown')),'decision_threshold':THRESHOLD,'feature_count':30,'test_metrics':card.get('selected_test_metrics',{})}
@app.get('/stats')
def stats():
 if not cache:return {'predictions':0,'redis_enabled':False}
 try:return {'predictions':int(cache.get('predictions') or 0),'redis_enabled':True}
 except redis.RedisError:return {'predictions':0,'redis_enabled':True,'error':'Redis unavailable'}
