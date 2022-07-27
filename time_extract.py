DATA_DIR='data/language/medcat_timelines/'
RESULT_SAVE='data/language/sutime_results/'
chunk_size=50000

import json
from sutime import SUTime
from datetime import datetime
import pandas as pd
import re, os, psutil, pickle, glob
import time
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
from utils import read_data, load_threads, timex2dt
import random

def infer_date_column(chunk_df):
    sutime = SUTime(jvm_started=False,mark_time_ranges=False, include_range=False)
    
    dt_list=[]
    for idx,row in chunk_df.iterrows():
        text=row['clean_text']
        rdate=row['created_at']
        rdatetime=datetime.strptime(rdate,"%Y-%m-%d")
        dates_js=sutime.parse(text,reference_date=rdate)
        if len(dates_js):
            dates=[dj['value'] for dj in dates_js if (dj['type'] in ['DATE','TIME'] and 'value' in list(dj.keys()))]
            dates=timex2dt(dates,FORMAT)
            dates=[dt for dt,error in dates if dt>=datetime(2019,1,1) and dt<=rdatetime]
            dt_list.append((row['id'],pd.Series(dates,dtype='datetime64[ns]').mean()))
        else:
            dt_list.append((row['id'],np.nan))
    return pd.DataFrame(dt_list,columns=['id','SUtime_date'])


if __name__ == '__main__':
    tw_files=os.listdir(DATA_DIR)
    tw_files.sort()
    tw_paths=[DATA_DIR+'/'+fi for fi in tw_files if fi[-4:]=='quet']
    
    df=load_threads(tw_paths)
    df['created_at']=df['created_at'].dt.strftime('%Y-%m-%d')
    df.dropna(subset=['id'],inplace=True)
    print('Found {} messages from {} users'.format(df.shape[0],df['user.id'].nunique()))

    df.sort_values('created_at',inplace=True)
    df.drop_duplicates(subset=['user.id','text'],inplace=True,keep='first')
    df.drop_duplicates(subset=['user.id','clean_text'],inplace=True,keep='first')
    print('Found {} messages from {} users'.format(df.shape[0],df['user.id'].nunique()))

    FORMAT = '%Y-%m-%d'
    
    df_chunks = [df[i:i+chunk_size] for i in range(0,df.shape[0],chunk_size)]
    print('DataFrame split into {} chunks\n'.format(len(df_chunks)))
    
    random.seed(721)
    s_t=time.time()
    pool=mp.Pool(2)
    df_list = tqdm(pool.imap(infer_date_column,df_chunks),total=len(df_chunks))
    pool.close()
    pool.join()
    df_list= pd.concat(df_list)
    e_t=time.time()-s_t
    print('Elapsed time: {} hrs'.format(e_t/60/60))
    
    if not df_list.shape[0]==df.shape[0]:
        print('Number of dates is not equal to number of tweets')
    
    df=df.merge(df_list,left_on='id',right_on='id')
    print(df.columns)
    time.sleep(10)
    print('Found {} messages from {} users'.format(df.shape[0],df['user.id'].nunique()))
    print('Counts:\n',df[['id','text','clean_text','created_at','SUtime_date']].count())

    print('Uniques:\n',df[['id','text','clean_text','created_at']].nunique())
    df.reset_index(inplace=True,drop=True)
    df['created_at']=pd.to_datetime(df['created_at'])
    print(df.shape)    
    with open(RESULT_SAVE+'sutime_df_newfilter.pkl','wb') as f:
        pickle.dump(df,f)
