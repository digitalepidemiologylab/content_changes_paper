### Final step of the pipeline -> combines all tweets with the annotations

import pandas as pd
import re, os, pickle, random, psutil
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from utils import load_threads, clean_tweet, clean_crowdbreaks

from filters import POSITIVE_FILTER

DATA_DIR='data/timelines/parsed/'
RESULT_SAVE='data/language/all_timelines/'
SAVE_USER_TIMELINES=True
USE_CACHE=False
INTEGRATE_LIST=False
emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
        "love", "optimism", "pessimism", "sadness", "surprise", 
        "trust", "neutral"]


def read_data(f_name,col_names=['created_at','id','user.id','text','lang']):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

def filter_user(parquet_file):
    user=parquet_file.split('/')[-1].split('.')[0]
    df=read_data(parquet_file,col_names=col_names)
    df['id']=df['id'].astype(str)
    
    df=pd.concat([df,df_pos[df_pos['user.id']==user]])
    df.loc[df['positive'].isnull(),'positive']=0
    if df['positive'].sum()<1:
        print('User {} is missing a positive tweet (with {} others)'.format(user,len(not_positive_users)))
        not_positive_users.append(user)
        
    try:
        df_url=read_data('data/language/url_timelines/'+user+'.parquet',col_names=url_col_names)
        df_url.id=df_url.id.astype(str)
    except FileNotFoundError:
        df_url=pd.DataFrame(columns=['id'])
    try:
        df_top=read_data('data/language/topic_timelines/'+user+'.parquet',col_names=topic_col_names)
        df_top.id=df_top.id.astype(str)
    except FileNotFoundError:
        df_top=pd.DataFrame(columns=['id'])
    try:
        df_emo=read_data('SpanEmo/data/all_processed_tweets_updated/'+
                     user+'.parquet', col_names=emo_col_names)
        df_emo['id']=df_emo['ID'].astype(str)
        mlb = MultiLabelBinarizer(classes=emotion_labels)
        df_emo = pd.DataFrame(mlb.fit_transform(df_emo.label),
                   columns= mlb.classes_,
                   index=df_emo.id)
    except FileNotFoundError:
        df_emo=pd.DataFrame()
        
    df=df.merge(df_url,how='left',on='id')
    df=df.merge(df_top,how='left',on='id')
    df=df.merge(df_emo,how='left',left_on='id',right_index=True)
    df=df.merge(df_sym[['id']+list(df_sym.columns[5:])],how='left',on='id')
    df.drop_duplicates(subset=['id'],keep='last',inplace=True)
    df.loc[(df.positive==1)&(df['SUtime_date'].isnull()),'SUtime_date'] = df['created_at']
    df['created_at']=df.created_at.apply(lambda x: x.strftime('%Y-%m-%d'))
    df['SUtime_date']=df.SUtime_date.apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else np.nan)
    if SAVE_USER_TIMELINES and len(df):
        df.to_parquet(RESULT_SAVE+parquet_file.split('/')[-1].split('.')[0]+'.parquet',
                     allow_truncated_timestamps=True)
    return df.shape[0]

            
col_names=['id','text','created_at','user.id','lang','url','url_domain']
url_col_names=['id','U_cat1','U_cat1w','U_cat2','U_cat2w','D_cat1','D_cat1w','D_cat2','D_cat2w']
topic_col_names=["id","Politics & Government & Law", "Business & Industry", "Health", "Science & Mathematics", "Computers & Internet & Electronics", "Society & Culture", "Education & Reference", "Entertainment & Music", "Sport"]
emo_col_names=['ID','label']
emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
                      "love", "optimism", "pessimism", "sadness", "surprise", "trust","neutral"]
# Loading dataset of "I have tested positive" tweets
with open('data/df_positive_newfilter.pkl','rb') as f:
    df_pos=pickle.load(f)
df_pos['id']=df_pos['id'].astype(str)
df_pos['positive']=1
df_pos['lang']='en'
manager = mp.Manager()
not_positive_users = manager.list()
print('\n {} tweets loaded for: testing positive to COVID'.format(df_pos.shape[0]))

# Loading dataset of symptoms tweets -> adding SUtime column -> keeping only self-reports
with open('data/language/sutime_results/sutime_df_newfilter.pkl','rb') as f:
    df_sym=pickle.load(f)
df_sym.loc[df_sym['SUtime_date'].isnull(),'SUtime_date'] = df_sym['created_at']
df_sym['SUtime_date']=pd.to_datetime(df_sym['SUtime_date'])
vstart=pd.to_datetime('2020-12-16',utc=0)
df_sym.loc[df_sym.SUtime_date<vstart,'vaccinated']=0
PRED_SR='reporting_classification/data/data_after_postprocessing/predictions/'
df_sr=pd.read_parquet(PRED_SR+'ids_text_filtered_predictions.parquet')
df_sr['self_prob']=df_sr['label_probabilities'].apply(lambda x:x['Self_reports'])
df_sr=df_sr[df_sr.self_prob>.9]
print('\n {} tweets loaded for: self-reports of symptoms predictions'.format(df_sr.shape[0]))
df_sym=df_sym.merge(df_sr[['id','self_prob']],how='left',on='id')
df_sym=pd.concat([df_sym[df_sym.vaccinated==1],df_sym[df_sym.positive==1],df_sym.dropna(subset=['self_prob'])])
df_sym.drop_duplicates('id',inplace=True)
print('\n {} tweets loaded for: self-reports of symptoms + vaccinated + positive'.format(df_sym.shape[0]))
print('{} positive users in symptoms dataset'.format(df_sym[df_sym.positive==1]['user.id'].nunique()))
if df_sym[df_sym.positive==1].shape[0] != df_sym[df_sym.positive==1]['SUtime_date'].count():
    print('\nWARNING: Positive dates missing:\n{} positive tweets in symptoms dataset'.format(df_sym[df_sym.positive==1].shape[0]))
    print('{} positive users with SUtime'.format(df_sym[df_sym.positive==1]['SUtime_date'].count()))
df_sym.drop(['self_prob','positive'],axis=1,inplace=True)


tw_files=os.listdir(DATA_DIR)
tw_files.sort()
users=[fi.split('.')[0] for fi in tw_files]
n_users=len(users)
print('{} total users.\n'.format(n_users))    


if USE_CACHE==True:
    res_files=os.listdir(RESULT_SAVE)
    res_users=set([fi.split('.')[0] for fi in res_files if fi[-4:]=='quet'])
    users=[fi for fi in users if fi not in res_users]
    print('{} users already computed parsed in the folder'.format(len(res_users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
if INTEGRATE_LIST==True:
    print('Parsing only a list of restored users...')
    with open('data/u_down_last.pkl','rb') as f:
        users_restored=pickle.load(f)
    users=list(set(users_restored).intersection(set(users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    
tw_paths=[DATA_DIR+us+'.parquet' for us in users]

s_t=time.time()
pool=mp.Pool(30)
res = list(tqdm(pool.imap(filter_user, tw_paths), total=n_users))
pool.close()
pool.join()
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages'.format(sum(res)))
