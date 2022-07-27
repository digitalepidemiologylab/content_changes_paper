### Convert timelines from .jsonl to DataFrame in .parquet, one for each user

import json, os, pickle
from tqdm import tqdm
import pandas as pd
import time
import multiprocessing as mp
import random
import numpy as np

DATA_DIR='data/timelines/raw/'
RESULT_SAVE='data/timelines/parsed/'
USE_CACHE=False
INTEGRATE_LIST=False

def parse_jsonl(f):
    try:
        temp= pd.read_json(f, lines=True)
    except ValueError as er:
        return (0,0)
    df=pd.DataFrame()
    df['created_at']=pd.DatetimeIndex(pd.to_datetime(temp.created_at,utc=True)).tz_convert('UTC')
    df['id']=[str(i) for i in temp.id.values]
    df['text']=temp.text
    df['user.id']=str(temp['author_id'].iloc[0]) 
    df['lang']=temp.lang
    df['user.screen_name']=[u['username'] for u in temp.author.values]
    df['url']=''
    df['url_domain']=''
    if 'entities' in temp.columns:
        temp.loc[temp.entities.isna(),'entities']=[{}]*temp.entities.isna().sum()
        df['url']=[random.choice(e['urls'])['expanded_url'] if 'urls' in e.keys() else '' for e in temp.entities.values]
        df['url_domain']=['https://'+u.split('/')[2] if u!='' else '' for u in df.url.values]
    df.drop_duplicates(subset=['id'],inplace=True)
    df.to_parquet(RESULT_SAVE+f.split('/')[-1].split('.')[0]+'.parquet')
    return (df.shape[0],1)

tw_files=os.listdir(DATA_DIR)
tw_files.sort()
users=[fi.split('.')[0] for fi in tw_files]
n_users=len(users)

if USE_CACHE==True:
    print('Checking cached documents...')
    res_files=os.listdir(RESULT_SAVE)
    res_users=set([fi.split('.')[0] for fi in res_files])
    users=[fi for fi in users if fi not in res_users]
    print('{} users already computed parsed in the folder'.format(len(res_users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
if INTEGRATE_LIST==True:
    print('Parsing only a list of restored users...')
    with open('data/u_down_ .pkl','rb') as f:
        users=pickle.load(f)
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
tw_paths=[DATA_DIR+us+'.jsonl' for us in users]
s_t=time.time()
pool=mp.Pool(32)
res = list(tqdm(pool.imap(parse_jsonl, tw_paths), total=n_users))
pool.close()
pool.join()
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages by {} users'.format(sum([r[0] for r in res]),sum([r[1] for r in res])))
