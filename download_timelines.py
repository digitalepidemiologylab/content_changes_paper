# Download the timelines for the users in the DataFrame -> .jsonl for each user.id

from twarc import Twarc2, expansions
import json, os, pickle
import configparser as CFG
from tqdm import tqdm
import pandas as pd
import datetime

START_DAY= datetime.datetime(2020, 1, 1, 0, 0, 0, 0, datetime.timezone.utc)
END_DAY= datetime.datetime(2021, 9, 29, 23, 59, 0, 0, datetime.timezone.utc)

RESULT_SAVE='data/timelines/raw/'
USE_CACHE=True

def get_timelines(client,users):
    # This timeline functions gets the Tweet timeline for a specified user
    for user in tqdm(users):
        user_timeline = client.timeline(user=user,start_time=START_DAY,end_time=END_DAY,
                                        exclude_retweets=True)

        # Twarc returns all Tweets for the criteria set above, so we page through the results
        for page in user_timeline:
            # The Twitter API v2 returns the Tweet information and the user, media etc.  separately
            # so we use expansions.flatten to get all the information in a single JSON
            result = expansions.flatten(page)
            for tweet in result:
                # Here we are printing the full Tweet object JSON to the console
                with open(RESULT_SAVE+str(user)+'.jsonl', 'a+') as f:
                        f.write(json.dumps(tweet) + '\n')



    
    
### read configuration
cfg = CFG.RawConfigParser()
cfg.read("settings.cfg")
file=cfg['track']['file']
if os.path.isfile(file):
    print('Selecting users from DataFrame\n')
    with open(file,'rb') as f:
        df_pos=pickle.load(f)
    users=list(df_pos['user.id'].unique())
else:
    print('Selecting users from .cfg file\n')
    # Tags are written in multiline. Use indents for that
    users = [int(c) for c in cfg['track']['users'].split("\n") if c != ""]
    
if USE_CACHE==True:
    res_files=os.listdir(RESULT_SAVE)
    res_users=set([fi.split('.')[0] for fi in res_files])
    users=[fi for fi in users if fi not in res_users]
    print('{} users already computed parsed in the folder'.format(len(res_users)))
    n_users=len(users)
    print('{} still to compute'.format(n_users))
    

# these are the credentials for the twitter api.
bearer_token = cfg["credentials"]["bearer"]

client = Twarc2(bearer_token=bearer_token)

get_timelines(client,users)
