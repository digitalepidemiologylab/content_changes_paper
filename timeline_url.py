import URLcategorization.config as config
from URLcategorization.functions import scrape_url
import pandas as pd
import re, os, pickle, random, psutil
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from utils import load_threads, clean_tweet, clean_crowdbreaks


with open(config.WORDS_FREQUENCY_PATH, "rb") as pickle_in:
    words_frequency = pickle.load(pickle_in)

DATA_DIR='data/timelines/parsed/'
RESULT_SAVE='data/language/'
SAVE_USER_TIMELINES=True
SAVE_CATEGORY_DICT=True
USE_CACHE=True


def read_data(f_name):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

def filter_user(parquet_file):
    """Find tweets about being vaccinated"""
    print('\n\n\nDomains:',len(dcats))
    user=parquet_file.split('/')[-1].split('.')[0]
    df=read_data(parquet_file)
    df=df.loc[df['lang']=='en']
    df.drop_duplicates(subset=['id'],keep='first',inplace=True)
    df=df[df.url!='']

    df[['U_cat1','U_cat1w','U_cat2','U_cat2w','D_cat1','D_cat1w','D_cat2','D_cat2w']]=np.nan
    for idx,row in df.iterrows():
        url = row['url']
        domain= row['url_domain']
        if domain!='https://twitter.com' and domain!='https://t.co':
            try:
                u_cat=ucats[url]
            except KeyError:
                u_cat=scrape_url(url,words_frequency)
                ucats[url]=u_cat
            try:
                d_cat=dcats[domain]
            except KeyError:
                d_cat=scrape_url(domain,words_frequency)
                dcats[domain]=d_cat
        else:
            u_cat=(np.nan,np.nan,np.nan,np.nan)
            d_cat=('Twitter',5e7,'Twitter',5e7)
        df.loc[idx,['U_cat1','U_cat1w','U_cat2','U_cat2w']]=u_cat
        df.loc[idx,['D_cat1','D_cat1w','D_cat2','D_cat2w']]=d_cat

    df.dropna(subset=['D_cat1'],inplace=True)
    if SAVE_CATEGORY_DICT:
        with open(RESULT_SAVE+'dcats.pkl','wb') as f:
            pickle.dump(dcats,f)
        with open(RESULT_SAVE+'ucats.pkl','wb') as f:
            pickle.dump(ucats,f)
    if SAVE_USER_TIMELINES and len(df):
        df[['id','url','U_cat1','U_cat1w','U_cat2','U_cat2w','url_domain','D_cat1','D_cat1w','D_cat2','D_cat2w']].to_parquet(RESULT_SAVE+'url_timelines/'+parquet_file.split('/')[-1].split('.')[0]+'.parquet')
    return df[['id','url','U_cat1','U_cat1w','U_cat2','U_cat2w','url_domain','D_cat1','D_cat1w','D_cat2','D_cat2w']]

            
col_names=['id','text','created_at','user.id','lang','url','url_domain']

with open(RESULT_SAVE+'dcats.pkl','rb') as f:
    dcats_old=pickle.load(f)
with open(RESULT_SAVE+'ucats.pkl','rb') as f:
    ucats_old=pickle.load(f)
    
manager = mp.Manager()
dcats=manager.dict()
ucats=manager.dict()
dcats.update(dcats_old)
ucats.update(ucats_old)

tw_files=os.listdir(DATA_DIR)
tw_files.sort()
users=[fi.split('.')[0] for fi in tw_files]
n_users=len(users)
print('{} total users.\n'.format(n_users))    


if USE_CACHE==True:
    res_files=os.listdir(RESULT_SAVE+'url_timelines/')
    res_users=set([fi.split('.')[0] for fi in res_files if fi[-4:]=='quet'])
    users=[fi for fi in users if fi not in res_users]
    print('{} users already computed parsed in the folder'.format(len(res_users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
tw_paths=[DATA_DIR+us+'.parquet' for us in users]

s_t=time.time()
pool=mp.Pool(16)
res = pd.concat(list(tqdm(pool.imap(filter_user, tw_paths), total=n_users)))
pool.close()
pool.join()
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages'.format(res.shape[0]),'\n')

print(res['U_cat1'].value_counts(),'\n')

print(res.columns)
