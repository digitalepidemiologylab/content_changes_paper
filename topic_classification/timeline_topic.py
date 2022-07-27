from transformers import pipeline
import pandas as pd
import re, os, pickle, random, psutil, csv
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import sys
from pyarrow.lib import ArrowInvalid
from utils import load_threads

DATA_DIR='data/ekphrasis_preprocessed/'
RESULT_SAVE='../data/language/'
SAVE_USER_TIMELINES=True
USE_CACHE=True
#Decide the cleaning function: run this script annotating both texts and print if see a difference


def read_data(f_name):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

        
def filter_user(parquet_file):
    """Find tweets about being vaccinated"""
    user=parquet_file.split('/')[-1].split('.')[0]
    try:
        df=read_data(parquet_file)
    except ArrowInvalid:
        return 0
    df.rename(columns={'ID':'id','Tweet':'text'},inplace=True)
    df.drop_duplicates(subset=['id'],keep='first',inplace=True)
    df=df[df.preprocessed_text!='']
    df[candidate_labels]=np.nan
    preds=nlp(list(df.preprocessed_text.values), candidate_labels, multi_label=True, 
              hypothesis_template=hypothesis_template)
    
    for i,(idx, row) in enumerate(df.iterrows()):
        labs=preds[i]['labels']
        df.loc[idx,labs]=preds[i]['scores']
    df.dropna(subset=['Health'],inplace=True)
    if SAVE_USER_TIMELINES and len(df):
        df.to_parquet(RESULT_SAVE+'topic_timelines/'+user+'.parquet')
    return len(df)

            
col_names=['ID','Tweet','preprocessed_text']

nlp = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3",device=1,num_workers=4, 
               batch_size=1024)
candidate_labels=["Politics & Government & Law", "Business & Industry", "Health", "Science & Mathematics", "Computers & Internet & Electronics", "Society & Culture", "Education & Reference", "Entertainment & Music", "Sport"]
hypothesis_template = "This text is about {}."

tw_files=os.listdir(DATA_DIR)
tw_files.sort()
users=[fi.split('.')[0] for fi in tw_files][20000:]
n_users=len(users)
print('{} total users.\n'.format(n_users))    


if USE_CACHE==True:
    res_files=os.listdir(RESULT_SAVE+'topic_timelines/')
    res_users=set([fi.split('.')[0] for fi in res_files if fi[-4:]=='quet' and fi.split('.')[0] in users])
    users=[fi for fi in users if fi not in res_users]
    print('{} users already computed parsed in the folder'.format(len(res_users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
tw_paths=[DATA_DIR+us+'.parquet' for us in users]

s_t=time.time()
res=[]
for twp in tqdm(tw_paths):
    res.append(filter_user(twp))
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages'.format(sum(res)),'\n')
