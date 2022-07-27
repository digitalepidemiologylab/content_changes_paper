import os
import joblib
from tqdm import tqdm

import time
import datetime
import pandas as pd
import numpy as np
import json

from causalimpact import CausalImpact
import random
import math

from pyarrow.lib import ArrowInvalid

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'language', 'all_timelines')

sym_cols = ['vaccinated', 'Fatigue', 'Malaise', 'Dyspnea', 'Chest Pain', 'atypical chest pain', 'acute chest pain',
        'burn chest', 'Fever', 'Coughing', 'Heartburn', 'Headache', 'Sore Throat', 'Nausea', 'Vomiting',
        'Dizziness', 'Myalgia', 'gastrointestinal discomfort', 'Audible crepitus of joint']
col_names = ['id', 'text', 'created_at', 'user.id', 'lang', 'url', 'url_domain', 'positive', 'SUtime_date']
url_col_names = ['U_cat1', 'U_cat1w', 'U_cat2', 'U_cat2w', 'D_cat1', 'D_cat1w', 'D_cat2', 'D_cat2w']
topic_col_names = ["Politics & Government & Law", "Business & Industry", "Health", 
        "Science & Mathematics", "Computers & Internet & Electronics", "Society & Culture", 
        "Education & Reference", "Entertainment & Music", "Sport"]

emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
                "love", "optimism", "pessimism", "sadness", "surprise", 
                "trust", "neutral"]
cols = col_names + emotion_labels + topic_col_names + url_col_names + sym_cols

use_cache = True
num_excluded_weeks = 0


def read_data(f_path, nweeks=12):
    try:
        df = pd.read_parquet(f_path)
    except ArrowInvalid:
        df = pd.DataFrame(columns=cols)
        return df
    
    wrong_cols = [col for col in df.columns if col not in cols]
    df.drop(wrong_cols, axis=1, inplace=True)
    df['created_at'] = pd.DatetimeIndex(pd.to_datetime(df.created_at, utc=True)).tz_convert(None)
    df['SUtime_date'] = pd.DatetimeIndex(pd.to_datetime(df.SUtime_date, utc=True)).tz_convert(None)

    # Loading parquets could change the user.id column if not seen as string, so we forcefully set it to the file name (user id)
    user = os.path.basename(f_path).split('.')[0]
    df_pos = df[df.positive==1]
    if not len(df_pos):
        return pd.DataFrame(columns=cols)
    user_pos = df_pos['user.id'].iloc[0]
    assert user_pos == user
    df['user.id'] = user_pos
    df['user.id'] = df['user.id'].astype(str)

    pdate = df_pos['SUtime_date'].min()
    df['pdate'] = pdate
    df['effective_date'] = df['SUtime_date']
    df.loc[df['SUtime_date'].isna(),'effective_date'] = df.loc[df['SUtime_date'].isna(),'created_at']

    df.drop(df[df['effective_date']<pd.to_datetime('2020-01-01')].index,axis=0,inplace=True)
    df.drop(df[df['effective_date']>=pd.to_datetime('2021-10-01')].index,axis=0,inplace=True)
    df['rel_effective_day'] = (df['effective_date']-df['pdate']).apply(lambda x: x.days)
    df['rel_effective_week'] = 0
    df.loc[df['rel_effective_day']<0,'rel_effective_week'] = (df.loc[df['rel_effective_day']<0,'rel_effective_day']/7).apply(math.ceil)
    df.loc[df['rel_effective_day']>0,'rel_effective_week'] = (df.loc[df['rel_effective_day']>0,'rel_effective_day']/7).apply(math.floor)
    df['rel_effective_month'] = 0
    df.loc[df['rel_effective_day']<0,'rel_effective_month'] = (df.loc[df['rel_effective_day']<0,'rel_effective_day']/30).apply(math.ceil)
    df.loc[df['rel_effective_day']>0,'rel_effective_month'] = (df.loc[df['rel_effective_day']>0,'rel_effective_day']/30).apply(math.floor)

    idate = pdate-pd.DateOffset(weeks=nweeks)
    edate = pdate+pd.DateOffset(weeks=nweeks)
    if edate >= pd.to_datetime('2021-10-01'):
        return pd.DataFrame(columns=cols)
    if idate < pd.to_datetime('2020-01-01'):
        return pd.DataFrame(columns=cols)
    df = df.loc[(df['effective_date']>=idate)&(df['effective_date']<edate)]
    vdates = pd.to_datetime(df[df['vaccinated']==1]['effective_date'].values)
    usymps = df[df[sym_cols].sum(1)>=1]
    sdates = usymps['effective_date'].values
    if pd.isna(pdate):
        return pd.DataFrame(columns=cols)
    adv_reacts = []
    for idx, row in usymps.iterrows():
        sdate = row['effective_date']
        twid = row['id']
        adv_react = False
        for vdate in vdates:
            if (sdate-vdate).days>=0 and (sdate-vdate).days<=8:
                adv_react=True
                adv_reacts.append(twid)
                break

    df.loc[df['id'].isin(adv_reacts),sym_cols] = 0

    return df.loc[(df.lang=='en')|(df.positive==1)]


def load_threads(f_paths, nweeks=12):
    ts = time.time()
    parallel = joblib.Parallel(n_jobs=30, prefer='threads')
    read_data_delayed = joblib.delayed(read_data)
    res = parallel(read_data_delayed(f_path, nweeks) for f_path in tqdm(f_paths, desc='Reading raw data and adding new columns (e.g., SUtime_date, pdate, rel_effective_week)'))
    df = pd.concat(res)
    te = time.time()
    logger.info(f'Data loaded in {te-ts:.5f} seconds')

    return df


def load_data(sample_size=0):
    tw_files = os.listdir(data_dir)
    tw_files.sort()
    if sample_size > 0:
        random.seed(723)
        tw_paths = random.sample([os.path.join(data_dir, f_name) for f_name in tw_files if f_name.split('.')[1] == 'parquet'], k=sample_size)
    else:
        tw_paths = [os.path.join(data_dir, f_name) for f_name in tw_files if f_name.split('.')[1] == 'parquet']

    df = load_threads(tw_paths, nweeks=12)
    
    logger.info(f'Found {df.shape[0]} messages from {df["user.id"].nunique()} users')

    return df


def preprocessing(df):
    features_cols = sym_cols + url_col_names + topic_col_names + emotion_labels
    df.drop_duplicates(subset=['id'],inplace=True)
    # Save rows with positive==1 to be concatenated after the NaN dropping
    positivity_tweets = df.loc[(df.positive == 1) & (df.rel_effective_day == 0)].copy()
    positivity_tweets = positivity_tweets.groupby('user.id').apply(lambda x: (x.sort_values(by='SUtime_date', ascending=True)).iloc[0]).reset_index(drop=True)
    positivity_tweets.dropna(subset=emotion_labels, how='all', inplace=True)
    # Drop rows where all values in features_cols are NaN
    df.dropna(subset=features_cols, how='all', inplace=True)
    df.dropna(subset=emotion_labels, how='all', inplace=True)
    df.dropna(subset=topic_col_names, how='all', inplace=True)
    # Add positive tweets back and remove duplicates
    df = pd.concat([df,positivity_tweets])
    df.drop_duplicates(subset=['id'],inplace=True)
    df[features_cols] = df[features_cols].fillna(0)
    df.sort_values(['user.id','SUtime_date'],inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_num_weeks(df, min_tweets_number_period=10):
    # Pre-period
    df_pre = df[df.rel_effective_week < 0].copy()
    # Only consider users who wrote at least 10 tweets in the pre-period (10 is the default value)
    min_tweets_per_user_pre_bool = df_pre.groupby('user.id')['id'].count() >= min_tweets_number_period
    remaining_users_pre = (min_tweets_per_user_pre_bool[min_tweets_per_user_pre_bool]).index.tolist()
    df_pre = df_pre.loc[df_pre['user.id'].isin(remaining_users_pre)]
    # Number of weeks corresponding to the pre-period for each user
    num_weeks_pre = df_pre.groupby('user.id')['rel_effective_week'].min().abs()
    # Keep users who posted tweets over 5 weeks or more *before* they reported their positivity to Covid
    num_weeks_pre = num_weeks_pre[num_weeks_pre >= 5]

    # Post-period
    df_post = df[df.rel_effective_week >= 0].copy()
    # Only consider users who wrote at least 10 tweets in the post-period (10 is the default value)
    min_tweets_per_user_post_bool = df_post.groupby('user.id')['id'].count() >= min_tweets_number_period
    remaining_users_post = (min_tweets_per_user_post_bool[min_tweets_per_user_post_bool]).index.tolist()
    df_post = df_post.loc[df_post['user.id'].isin(remaining_users_post)]
    
    # Number of weeks corresponding to the post-period for each user
    num_weeks_post = df_post.groupby('user.id')['rel_effective_week'].max().abs()
    # Keep users who posted tweets over 5 weeks or more *after* they reported their positivity to Covid
    num_weeks_post = num_weeks_post[num_weeks_post >= 5]
    # Intersection of users
    users_pre = set(num_weeks_pre.index)
    users_post = set(num_weeks_post.index)
    selected_users = list(users_pre.intersection(users_post))
    # Keep selected users
    num_weeks_pre = num_weeks_pre[selected_users]
    num_weeks_post = num_weeks_post[selected_users]
    return num_weeks_pre, num_weeks_post, selected_users


def onehot_encoding_topics(df):
    def bool_cols(*topics_cols):
        # Labels with a probability of at least 50% are kept as valid topic tags
        bool_series = [v >= 0.5 for v in topics_cols][0]
        y = {col_name: bool_v for col_name, bool_v in zip(bool_series.index, bool_series)}
        names_cols = [k for k, v in y.items() if v]
        if names_cols == []:
            names_cols = ['other']
        return names_cols
    df['topic'] = df.apply(lambda x: bool_cols(x[topic_col_names]), axis=1)
    
    # Binary encoding
    for feature in [*topic_col_names, 'other']:
        df[f'is_{feature}'] = df.topic.apply(lambda x: 1 if feature in x else 0)

    return df


def onehot_encoding_urls(df):
    col_onehot = 'U_cat1'
    df[col_onehot] = df[col_onehot].replace(to_replace='Twitter', value=0)
    df[col_onehot] = df[col_onehot].replace(to_replace=0, value='undefined')
    df[col_onehot] = df[col_onehot].replace(to_replace='0', value='undefined')
    df[col_onehot] = df[col_onehot].apply(lambda x: col_onehot + '_' + x)
    df = pd.concat([df, pd.get_dummies(df[col_onehot])], axis=1)
    existing_cols = df.columns.tolist()
    new_cols = [col for col in existing_cols if col.startswith(col_onehot+'_')]

    return df, new_cols


def pre_post_analysis_single_user(user_df, significance_level=0.05):
    user_id = user_df['user.id'].unique()[0]
    output_df = pd.DataFrame(index=[user_id])
    output_df.index.rename('user.id', inplace=True)

    min_week_user = user_df.rel_effective_week.min()
    max_week_user = user_df.rel_effective_week.max()
    all_weeks = np.arange(min_week_user, max_week_user+1).tolist()
    week0_position = all_weeks.index(0)
    pre_period = [0, week0_position - 1]
    post_period = [week0_position, len(all_weeks)-1]
    
    feature = 'full_volume'
    if feature == 'full_volume':
        available_weeks_counts = user_df.groupby('rel_effective_week')['id'].count()
    else:
        available_weeks_counts = user_df.groupby('rel_effective_week')[f'is_{feature}'].sum()
    
    available_weeks = set(available_weeks_counts.index)
    missing_weeks = list(set(all_weeks) - available_weeks)
    missing_weeks_counts = pd.Series([0]* len(missing_weeks))
    missing_weeks_counts.index = list(missing_weeks)
    weekly_rates_user = pd.concat([available_weeks_counts, missing_weeks_counts]).sort_index()
    data = weekly_rates_user.reset_index(drop=True)
    ci = CausalImpact(data, pre_period, post_period)
    p_value = ci.p_value
    null_rejected = p_value < significance_level
    output_df.loc[user_id, f'{feature}_pvalue'] = p_value
    output_df.loc[user_id, f'{feature}_null_rejected'] = null_rejected
    if null_rejected:
        abs_effect_avg = ci.summary_data.loc['abs_effect', 'average'] 
        if abs_effect_avg < 0:
            output_df.loc[user_id, f'{feature}_pre_to_post'] = 'decrease'
        elif abs_effect_avg > 0:
            output_df.loc[user_id, f'{feature}_pre_to_post'] = 'increase'
    else:
        output_df.loc[user_id, f'{feature}_pre_to_post'] = 'no_change' 

    return output_df


def test_feature_rate(df_selected_users, feature, selected_users, info_pre_and_post_periods, collective_analysis_dir):
    count_feature_pre = df_selected_users[df_selected_users.rel_effective_week < 0].groupby('user.id')[feature].sum()
    count_feature_pre = count_feature_pre[selected_users]
    fraction_feature_pre = count_feature_pre.div(info_pre_and_post_periods.count_pre_positivity)

    count_feature_post = df_selected_users[df_selected_users.rel_effective_week >=num_excluded_weeks].groupby('user.id')[feature].sum()
    count_feature_post = count_feature_post[selected_users]
    fraction_feature_post = count_feature_post.div(info_pre_and_post_periods.count_post_positivity)
    print(f'Feature: {feature}')

    fraction_df = pd.concat([fraction_feature_pre, fraction_feature_post], axis=1)
    fraction_df.columns = ['avg_pre', 'avg_post']
    fraction_df.to_csv(os.path.join(collective_analysis_dir, f'{feature}_pre_post_fractions.csv'))
    return


def generate_info_individual_results(individuals_df, selected_users):
    num_unchanged = len(individuals_df.loc[individuals_df.full_volume_pre_to_post == 'no_change'])
    num_increased = len(individuals_df.loc[individuals_df.full_volume_pre_to_post == 'increase'])
    num_decreased = len(individuals_df.loc[individuals_df.full_volume_pre_to_post == 'decrease'])
    num_total = len(individuals_df)
    summary_dict = {
            'num_unchanged': num_unchanged, 'num_increased': num_increased, 'num_decreased': num_decreased, 
            'frac_unchanged': num_unchanged/num_total,'frac_increased': num_increased/num_total,'frac_decreased': num_decreased/num_total,
            'num_selected_users': num_total, 'selected_users': selected_users
            }
    return summary_dict 


def remove_tweets_with_media_mentions(positivity_tweets):
    media_df = pd.read_csv('english_speaking_media.csv')
    media_list = media_df.username.tolist()
    media_list_str = '|'.join(media_list)
    positivity_tweets = positivity_tweets.loc[~positivity_tweets.text.str.contains(media_list_str)]
    # Add symbol in front of username
    ref_to_media_list = ['@'+ username for username in media_list]
    ref_to_media_list_str = '|'.join(ref_to_media_list)
    positivity_tweets = positivity_tweets.loc[~positivity_tweets.text.str.contains(ref_to_media_list_str)]
    return positivity_tweets


def main():
    time_fmt = '%Y-%m-%d_%H-%M-%S'
    # Create output folders
    results_dir = os.path.join('results_statistical_analysis', f'run') 
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    individual_analysis_dir = os.path.join(results_dir, 'individual_tests')
    if not os.path.isdir(individual_analysis_dir):
        os.makedirs(individual_analysis_dir)

    collective_analysis_dir = os.path.join(results_dir, 'collective_tests')
    if not os.path.isdir(collective_analysis_dir):
        os.makedirs(collective_analysis_dir)
    
    logger.info('Load data...')

    if use_cache:
        df = pd.read_csv(os.path.join('results_statistical_analysis', 'preprocessed_data.csv'), dtype={'id': str, 'user.id': str, 'Sport': float, 'U_cat1w': str, 'U_cat2w': str, 'D_cat1w': str, 'Audible crepitus of joint': float})
    else:
        df = load_data()
        logger.info('Prepare data...')
        df = df.loc[(df['user.id'].notna()) & (df['user.id']!= 'en')]
        df['user.id'] = df['user.id'].astype(str)
        df = df.loc[~df['user.id'].isin(['en', 'nan'])]
        df = preprocessing(df)
        df.to_csv(os.path.join('results_statistical_analysis', 'preprocessed_data.csv'), index=False)

    # Filter positivity tweets; for each user we consider the earliest positive tweet (df may contain more than one positive tweet per user)
    positivity_tweets = df.loc[(df.positive == 1) & (df.rel_effective_day == 0)].copy()
    positivity_tweets = positivity_tweets.groupby('user.id').apply(lambda x: (x.sort_values(by='SUtime_date', ascending=True)).iloc[0])
    # Exclude tweets containing references to news media; the user may be quoting a media reporting that a public character tested positive to Covid
    positivity_tweets = remove_tweets_with_media_mentions(positivity_tweets)

    # Do not include the aforementioned positivity tweets for the individual- and collective-level analyses
    df = df.loc[(df['user.id'].isin(positivity_tweets['user.id'].tolist())) & (~df.id.isin(positivity_tweets.id.tolist()))]
    # Note that df still contains positivity tweets as long as these are not the ones in positivity_tweets
    # These positive tweets originate from users who reported that they were positive to Covid more than once.
    # df discards their earliest positivity report but retains positivity tweets that came later in time

    # Update positivity_tweets 
    positivity_tweets = positivity_tweets[positivity_tweets['user.id'].isin(df['user.id'].tolist())]
    # The above update is a sanity check to exclude users whose respective timelines 
    # would contain no tweet in addition to the single positivity tweet
    
    logger.info(f'Data prepared ({df["user.id"].nunique()} users)')

    # Get number of weeks in the pre- and post-periods, respectively, for each user selected
    min_tweets_number_period = 30
    logger.info(f'Select users with at least {min_tweets_number_period} tweets in each period (pre-period and post-period)...')
    num_weeks_pre, num_weeks_post, selected_users = get_num_weeks(df, min_tweets_number_period=min_tweets_number_period)

    # Keep data from selected users
    df_selected_users = df.loc[df['user.id'].isin(selected_users)].copy()
    logger.info(f'Sample composed of {len(selected_users)} users')


    # Add indicator variables based on topic probabilities
    logger.info('Apply one-hot encoding for the topics...')
    df_selected_users = onehot_encoding_topics(df_selected_users)
    
    df_selected_users, url_cols = onehot_encoding_urls(df_selected_users) 

    # Discard tweets from excluded weeks
    excluded_weeks = np.arange(num_excluded_weeks).tolist()
    df_selected_users = df_selected_users.loc[~df_selected_users.rel_effective_week.isin(excluded_weeks)]
    # Save the above DataFrame
    df_selected_users.to_csv(os.path.join(individual_analysis_dir, 'tweets_selected_users_pre_post_periods.csv'), index=False)
    # Update list of selected users
    selected_users = df_selected_users['user.id'].unique().tolist()

    # Filter positivity tweets for the selected users
    positivity_tweets_selected_users = positivity_tweets.loc[positivity_tweets['user.id'].isin(selected_users)].copy()
    # Save these tweets
    positivity_tweets_selected_users.to_csv(os.path.join(individual_analysis_dir, 'users_positivity_tweets_between_pre_and_post_periods.csv'), index=False) 

    num_weeks_pre = num_weeks_pre[selected_users]
    num_weeks_post = num_weeks_post[selected_users]
    # As data from excluded weeks will not be taken into account, we should subtract the corresponding number of weeks from the values in num_weeks_post
    num_weeks_post = num_weeks_post - len(excluded_weeks)

    logger.info('Compute rates (i.e., how many tweets each user wrote each week in the pre- and post-period, respectively)...')
    # Estimated rate of tweets posted every week before the positivity tweet
    estimated_rate_pre = df_selected_users[df_selected_users.rel_effective_week<0].groupby('user.id')['id'].count().div(num_weeks_pre).dropna()
    estimated_rate_pre = estimated_rate_pre[selected_users]
    # Estimated rate of tweets in the post-period 
    estimated_rate_post = df_selected_users[df_selected_users.rel_effective_week>=num_excluded_weeks].groupby('user.id')['id'].count().div(num_weeks_post).dropna()
    estimated_rate_post = estimated_rate_post[selected_users]

    info_pre_and_post_periods = pd.concat([num_weeks_pre, num_weeks_post, estimated_rate_pre, estimated_rate_post], axis=1)
    info_pre_and_post_periods.columns = ['num_weeks_pre_positivity', 'num_weeks_post_positivity', 'estimated_rate_pre_positivity', 'estimated_rate_post_positivity']
    info_pre_and_post_periods['count_pre_positivity'] = info_pre_and_post_periods.estimated_rate_pre_positivity.mul(info_pre_and_post_periods.num_weeks_pre_positivity)
    info_pre_and_post_periods['count_post_positivity'] = info_pre_and_post_periods.estimated_rate_post_positivity.mul(info_pre_and_post_periods.num_weeks_post_positivity)

    ## Individual-level analysis ##
    logger.info('Starting individual-level analysis...')
    # Generate list of DataFrames used for parallelization
    user_df_list = []
    for user_id in selected_users:
        user_df = df_selected_users.loc[df_selected_users['user.id'] == user_id].copy()
        user_df_list.append(user_df)

    parallel = joblib.Parallel(n_jobs=24)
    user_analysis_delayed = joblib.delayed(pre_post_analysis_single_user)
    res = parallel(user_analysis_delayed(user_df) for user_df in tqdm(user_df_list, desc='Time series analysis for each user'))
    output_df = pd.concat(res)
    info_pre_and_post_periods = pd.concat([info_pre_and_post_periods, output_df], axis=1)
    # Save results
    info_pre_and_post_periods.to_csv(os.path.join(individual_analysis_dir, 'individual_results.csv'), index=False)
    logger.info('Individual-level analysis completed...')

    # Aggregate individual results
    summary_individual_results = generate_info_individual_results(info_pre_and_post_periods, selected_users)
    # Save results
    with open(os.path.join(individual_analysis_dir, 'pre_to_post_overall_changes.json'), 'w') as f:
        json.dump(summary_individual_results, f)
    

    ## Compute pre-post rates for collective statistical analysis
    rate_df = pd.concat([estimated_rate_pre, estimated_rate_post], axis=1)
    rate_df.columns = ['avg_pre', 'avg_post']
    rate_df.to_csv(os.path.join(collective_analysis_dir, 'full_volume_pre_post_rates.csv'))
    logger.info('Starting collective-level analysis on specific features...')
    topic_categories = ["is_"+topic for topic in topic_col_names]
    topic_categories = [*topic_categories, 'is_other']
    symptom_cols = ['Fatigue', 'Malaise', 'Dyspnea', 'Chest Pain', 'Fever', 'Coughing', 'Headache',
                    'Sore Throat', 'Nausea', 'Vomiting', 'Dizziness', 'Myalgia']

    collective_analysis_features = emotion_labels + topic_categories + url_cols + symptom_cols 
    logger.info('Compute pre/post rates...')
    for feature in collective_analysis_features:
        test_feature_rate(df_selected_users, feature, selected_users, info_pre_and_post_periods, collective_analysis_dir)
    
    logger.info('Done')


if __name__ == '__main__':
    main()
