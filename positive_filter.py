### Filter "I have tested positive" tweets from the COVID archive -> .parquet for each day

DATA_DIR='preprocess/data/1_parsed/tweets/'
RESULT_SAVE='data/positive/'
USE_CACHE=True

import pandas as pd
import re, os, psutil
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from utils import read_data, load_threads
from filters import POSITIVE_FILTER


def read_data(f_name):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

def test_positive(parquet_file):
    """Find tweets about being tested positive"""
    df=read_data(parquet_file).replace('nan',np.nan)
    df=df[df.retweeted_status_id.isnull()]
    df=df[df['lang']=='en']
    df.drop(['lang','retweeted_status_id'],axis=1,inplace=True)
    df=df[df.text.str.contains(POSITIVE_FILTER,case=False,na=False)]
    df[['id','text','created_at','user.id']].to_parquet(RESULT_SAVE+'pos_'+parquet_file.split('/')[-1][7:])
    return df.shape[0]

col_names=['created_at','id','user.id','text','retweeted_status_id','lang']
tw_files=os.listdir(DATA_DIR)

if USE_CACHE==True:
    res_files=os.listdir(RESULT_SAVE)
    res_days=set([fi[9:] for fi in res_files])
    tw_files=[fi for fi in tw_files if fi[7:] not in res_days]
    print('{} days already computed parsed in the folder'.format(len(res_days)))

n_days=len(tw_files)
print('{} still to compute'.format(n_days))

    
tw_files.sort()
tw_paths=[DATA_DIR+fi for fi in tw_files]

s_t=time.time()
pool=mp.Pool(16)
res = list(tqdm(pool.imap(test_positive, tw_paths[:n_days]), total=n_days))
pool.close()
pool.join()
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages'.format(sum(res)))
