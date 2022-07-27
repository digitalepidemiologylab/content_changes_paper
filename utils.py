import joblib, psutil, time, re, datetime, html, unicodedata
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
control_char_regex = re.compile(r'[\r\n\t]+')

def read_data(f_name):
    """Reads single parquet file"""
    df= pd.read_parquet(f_name)
    df['created_at']=pd.DatetimeIndex(pd.to_datetime(df.created_at,utc=True)).tz_convert('UTC')
    return df

def load_threads(f_names):
    """Load data with threads"""
    ts = time.time()
    parallel = joblib.Parallel(n_jobs=18, prefer='threads')
    read_data_delayed = joblib.delayed(read_data)
    res = parallel(read_data_delayed(f_name) for f_name in tqdm(f_names))
    df = pd.concat(res)
    te = time.time()
    print(f'Load threads took {te-ts:.5f} sec')
    return df

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp=tweet
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","@user", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '<url>', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = temp.split()
    temp = " ".join(word for word in temp)
    return temp

def clean_tfidf(tweet):
    if type(tweet) == np.float:
        return ""
    temp=tweet
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("#","", temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = temp.split()
    temp = " ".join(word for word in temp)
    temp = re.sub(url_regex, "", temp)
    temp = re.sub(username_regex, "", temp)  
    return temp.lower()



def clean_crowdbreaks(s):
        if not s:
            return ''
        if not isinstance(s, str):
            s = str(s)
        # convert HTML
        s = html.unescape(s)
        # replace \t, \n and \r characters by a whitespace
        s = re.sub(control_char_regex, ' ', s)
        # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
        s =  "".join(ch for ch in s if unicodedata.category(ch)[0] != 'C')
        # remove duplicate whitespace
        s = ' '.join(s.split())
        s = replace_urls(s)
        s = replace_usernames(s)
        return s
    
def replace_urls(text, filler='<url>'):
        # replace other urls by filler
        text = re.sub(url_regex, filler, str(text))
        # add spaces between, and remove double spaces again
        text = text.replace(filler, f' {filler} ')
        text = ' '.join(text.split())
        return text

def replace_usernames(text, filler='@user'):
        # replace other user handles by filler
        text = re.sub(username_regex, filler, str(text))
        # add spaces between, and remove double spaces again
        text = text.replace(filler, f' {filler} ')
        text = ' '.join(text.split())
        return text

def get_simple_date(item, strformat):
    try:
        return ('D',True, datetime.datetime.strptime(item[:10], strformat), strformat)
    except (ValueError, TypeError):
        return ('D',False, item, strformat)

def get_from_split(error,is_resolved, item, strformat):
    if is_resolved:
        return (error,is_resolved, item, strformat)
    try:
        tokens = item.split(' ')
        are_resolved, items = zip(*(get_simple_date(token, strformat) for token in tokens if 'INTERSECT' not in token))
        if any(are_resolved):
            # assume one valid token
            result, = (item for item in items if isinstance(item, datetime.datetime))
            return ('D',True, result, strformat)
    except (ValueError, AttributeError):
        pass
    return (error,False, item, strformat)

def get_from_no_day(error,is_resolved, item, strformat):
    if is_resolved:
        return (error,is_resolved, item)
    if not 'W' in item:
        try:
            mday=str(random.randint(1,28)).zfill(2)
            return ('M',True, datetime.datetime.strptime(f'{item[:7]}-{mday}', strformat))
        except ValueError:
            pass
    return (error,False, item)

def get_from_w_date(error,is_resolved, item):
    if is_resolved:
        return (error,is_resolved, item)
    if 'W' in item:
        wday=str(random.randint(0,6))
        try:
            if item[:4]=='2020':
                wk=int(item.split('W')[1][:2])-1
                new_item='2020-W'+str(wk)
                return ('W',True, datetime.datetime.strptime(f'{new_item[:8]}-{wday}', "%Y-W%W-%w"))
            
            elif item[:4]=='2021':
                wk=int(item.split('W')[1][:2])
                new_item='2021-W'+str(wk)
                return ('W',True, datetime.datetime.strptime(f'{new_item[:8]}-{wday}', "%Y-W%W-%w"))
            else:
                pass
        except ValueError as err:
                print('Error for:'+item)
                print(err)
                pass
    # If arrives there, is not resolved-> discarded in timex2dt (error not considered)
    return ('N',is_resolved, item)

def timex2dt(dates,strformat):
    collection1 = (get_simple_date(item,strformat) for item in dates)
    collection2 = (get_from_split(*args) for args in collection1)
    collection3 = (get_from_no_day(*args) for args in collection2)
    collection4 = (get_from_w_date(*args) for args in collection3)
    return [(d,error)  for error,is_resolved, d in collection4 if is_resolved ]
