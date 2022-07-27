import pandas as pd
import re, os, psutil, pickle, random
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from utils import read_data, load_threads, clean_tweet, clean_crowdbreaks
from medcat.vocab import Vocab
from medcat.cdb import CDB
from medcat.cat import CAT

from filters import POSITIVE_FILTER, VACCINE_FILTER

DATA_DIR='data/timelines/parsed/'
RESULT_SAVE='data/language/'
FINE_TUNE=False
LOAD_FINE_TUNED= not FINE_TUNE
SAVE_USER_TIMELINES=True
CLEAN_CROWDBREAKS=True
#Decide the cleaning function: run this script annotating both texts and print if see a difference

if CLEAN_CROWDBREAKS:
    clean_fun=clean_crowdbreaks


def read_data(f_name):
    """Reads single parquet file"""
    return pd.read_parquet(f_name,columns=col_names)

def filter_user(parquet_file):
    """Find tweets about being vaccinated"""
    user=parquet_file.split('/')[-1].split('.')[0]
    df_pos_us=df_pos[df_pos['user.id']==user]
    df=read_data(parquet_file)
    df=pd.concat([df.loc[df['lang']=='en',col_names[:-1]],df_pos_us],axis='rows')
    df.loc[df['positive'].isnull(),'positive']=0
    df.drop_duplicates(subset=['id'],keep='last',inplace=True)
    if df['positive'].sum()<1:
        print('User {} is missing a positive tweet (with {} others)'.format(user,len(not_positive_users)))
        not_positive_users.append(user)
    df['clean_text']=df['text'].apply(clean_fun)
    df['vaccinated']=0
    vac_rows=df.clean_text.str.contains(       
        VACCINE_FILTER,
        case=False,na=False)
    df.loc[vac_rows,'vaccinated']=1
    # Da fare-> salva dataframe solo con le colonne presenti per questo utente
    # poi nel concat di tutti gli utenti avremo tutte le colonne
    for idx,row in df.iterrows():
        doc = cat(row['clean_text'])
        if doc:
            for ent in doc.ents:
                sname=cdb.get_name(ent._.cui)
                df.loc[idx,sname]=1

    v_pos=list(df.columns).index('vaccinated')
    df=df[col_names[:-1]+['clean_text','positive']+list(df.columns)[v_pos:]]
    i_pos=list(df.columns).index('positive')
    df=df[df.iloc[:,i_pos:].sum(axis=1)>0]
    if SAVE_USER_TIMELINES:
        df[['id','text','clean_text','created_at','user.id']+list(df.columns)[i_pos:]].to_parquet(RESULT_SAVE+'medcat_timelines/'+parquet_file.split('/')[-1].split('.')[0]+'.parquet')
    return df[['id','text','clean_text','created_at','user.id','positive','vaccinated']+list(df.columns)[i_pos+2:]]

            
col_names=['id','text','created_at','user.id','lang']

with open('data/df_positive_newfilter.pkl','rb') as f:
    df_pos=pickle.load(f)
df_pos=df_pos[col_names[:-1]] #not have "lang"
df_pos['id']=df_pos['id'].astype(str)
df_pos['positive']=1
manager = mp.Manager()
not_positive_users = manager.list()



#Load MEDcat
vocab = Vocab.load(RESULT_SAVE+'vocab.dat')
if LOAD_FINE_TUNED:
    cdb = CDB.load(RESULT_SAVE+'fine_tuned.dat') 
    print('Load fine tuned model.')
else:
    # Load the cdb model you downloaded
    cdb = CDB.load(RESULT_SAVE+'cdb-medmen-v1_2.dat') 
    print('Load default model.')

#'T079' temporal concept
#'T047' disease or syndrome
#'T033' finding
tui_filter = ['T184']
cui_filters = set()
for tui in tui_filter:
    cui_filters.update(cdb.addl_info['type_id2cuis'][tui])

cdb.config.preprocessing['words_to_skip']=cdb.config.preprocessing['words_to_skip'].union(set(['dm','psn','disease','sign']))
cdb.config.ner['min_name_len'] = 4
cdb.config.linking['train_count_threshold'] = 10
cdb.config.linking['filters']['cuis'] = cui_filters

cdb.config.general['log_level']='INFO'

# Create cat - each cdb comes with a config that was used
#to train it. You can change that config in any way you want, before or after creating cat.
cat = CAT(cdb=cdb, config=cdb.config, vocab=vocab)
print('Skipping:',cat.config.preprocessing['words_to_skip'])


if FINE_TUNE:
    print('Fine-tuning...')
    # Self-supervised training
    df_pos['clean_text']=df_pos['text'].apply(clean_fun)
    cat.train(df_pos['clean_text'].values,progress_print=5000)
    cat.cdb.save(RESULT_SAVE+'fine_tuned.dat')
        

tw_files=os.listdir(DATA_DIR)
tw_files.sort()
users=[fi.split('.')[0] for fi in tw_files]
n_users=len(users)
    
tw_paths=[DATA_DIR+us+'.parquet' for us in users]

s_t=time.time()
pool=mp.Pool(16)
res = pd.concat(list(tqdm(pool.imap(filter_user, tw_paths), total=n_users)))
pool.close()
pool.join()
e_t=time.time()-s_t
print('Elapsed time:{} hrs'.format(e_t/60/60))
print('Found {} messages from {} users'.format(res.shape[0],res['user.id'].nunique()),'\n')

print('Positive:\nFound {} messages from {} users'.format(res[res.positive==1].shape[0],
                                                          res[res.positive==1]['user.id'].nunique()),'\n')

print('Vaccinated:\nFound {} messages from {} users'.format(res[res.vaccinated==1].shape[0],
                                                          res[res.vaccinated==1]['user.id'].nunique()),'\n')

print('{} users do not have a positive tweet'.format(len(not_positive_users)))

print(res.columns)

if CLEAN_CROWDBREAKS:
    with open(RESULT_SAVE+'df_CB_VS.pkl','wb') as f:
        pickle.dump(res,f)
else:
    with open(RESULT_SAVE+'df_VS.pkl','wb') as f:
        pickle.dump(res,f)
