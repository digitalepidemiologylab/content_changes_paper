import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from new_statistical_analysis import onehot_encoding_topics
import scipy.stats as st

topic_col_names = ["Politics & Government & Law", "Business & Industry", "Health", 
        "Science & Mathematics", "Computers & Internet & Electronics", "Society & Culture", 
        "Education & Reference", "Entertainment & Music", "Sport"]
emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
        "love", "optimism", "pessimism", "sadness", "surprise", 
        "trust", "neutral"]


def main():
    individual_dir = os.path.join('results_statistical_analysis', 'run', 'individual_tests')
    positive_tweets_fpath = os.path.join(individual_dir, 'users_positivity_tweets_between_pre_and_post_periods.csv')
    positive_tweets_df = pd.read_csv(positive_tweets_fpath)
    positive_tweets_df['effective_date'] = pd.to_datetime(positive_tweets_df.effective_date)
    positive_tweets_df.set_index('effective_date', inplace=True)
    
    # Monthly number of tweets
    monthly_counts_df = positive_tweets_df.groupby(pd.Grouper(freq='M'))['id'].count()
    monthly_counts_df.index = [dt.strftime('%B %y') for dt in monthly_counts_df.index.date]


    # Covid-19 cases
    cases = pd.read_csv('data/owid-covid-data.csv')
    cases = cases[cases['iso_code'].isin(['USA','GBR','AUS','CAN'])]
    cases['date'] = pd.to_datetime(cases.date)
    cases = cases.loc[(cases['date']>= positive_tweets_df.index.min()) & (cases['date']<= positive_tweets_df.index.max())]
    cases.set_index('date',inplace=True)
    monthly_cases = cases.groupby(pd.Grouper(freq='W'))['new_cases'].sum()
    monthly_cases.index = monthly_cases.index.date
    
    # Plot time at which positive tweets were written 
    fig_counts, ax_counts = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),dpi=300)
    ax_counts.bar(monthly_counts_df.index,monthly_counts_df.values, color='dodgerblue', edgecolor='black',
                  width=5, label='Tweets')
    ax_counts.set_xlabel('Time (end of week)')
    ax_counts.set_ylabel('Tweets count', color='dodgerblue')
    ax_counts.legend(loc='upper left')
    
    #Plot cases
    ax_cases = ax_counts.twinx()
    ax_cases.plot(monthly_cases.index,monthly_cases.values, color='darkseagreen', label='Cases')
    ax_cases.set_ylabel('New Covid-19 cases',color='darkseagreen')
    ax_cases.legend()
    
    fig_counts.tight_layout()
    fig_counts.savefig(os.path.join(individual_dir, 'weekly_counts_positive_tweets_cases.png'), dpi=300)
    
    print('Pearson correlation with lag:')
    for lag in range(-8,8):
        lag=int(lag)
        if lag<0:
            print(len(monthly_cases.iloc[:lag]))
            print(monthly_counts_df.shift(lag).dropna().shape[0])
            pcc, pval= st.pearsonr(monthly_cases.iloc[:lag],monthly_counts_df.shift(lag).dropna())
        else:
            print()
            print(len(monthly_cases.iloc[lag:]))
            print(monthly_counts_df.shift(lag).dropna().shape[0])
            pcc, pval= st.pearsonr(monthly_cases.iloc[lag:],monthly_counts_df.shift(lag).dropna())
        print(f'Lag:{lag}\tPearson:{pcc}\tpvalue:{pval}\n')
    
    # Topic-related columns contain probability values and it is more convenient to add analogous columns with one-hot encoded values
    positive_tweets_df = onehot_encoding_topics(positive_tweets_df)
    
    data = positive_tweets_df[emotion_labels].sum().sort_values()
    fsize1 = 25
    fsize2 = fsize1 - 5
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,15))
    ax.barh(data.index, data, color=['dodgerblue'],edgecolor='black')
    ax.set_xlabel('Count', fontsize=fsize1)
    ax.set_ylabel('Emotion', fontsize=fsize1)
    ax.tick_params(axis='x', labelsize=fsize2)
    ax.tick_params(axis='y', labelsize=fsize2)
    fig.savefig(os.path.join(individual_dir, 'distribution_emotions_in_positive_tweets.pdf'), dpi=300) 

    # Plot emotion counts on a monthly basis
    monthly_counts_emotions_df = positive_tweets_df.groupby(pd.Grouper(freq='M'))[emotion_labels].sum()
    monthly_counts_emotions_df.index = monthly_counts_emotions_df.index.date
    monthly_counts_emotions_df.to_csv(os.path.join(individual_dir, 'monthly_counts_emotions_positive_tweets.csv'))
    
    cmap = plt.get_cmap('tab20')
    fig_emotions1, ax_emotions1 = plt.subplots(nrows=1, ncols=1, figsize=(40,40))
    monthly_counts_emotions_df[emotion_labels].plot.bar(ax=ax_emotions1, stacked=True, colormap=cmap, edgecolor='black')
    ax_emotions1.legend(title='Emotion', loc='upper left', fontsize=fsize1)
    ax_emotions1.set_xlabel('Time', fontsize=fsize1)
    ax_emotions1.set_ylabel('Count', fontsize=fsize1)
    ax_emotions1.tick_params(axis='x', labelsize=fsize2)
    ax_emotions1.tick_params(axis='y', labelsize=fsize2)
    fig_emotions1 = ax_emotions1.get_figure()
    fig_emotions1.savefig(os.path.join(individual_dir, 'monthly_counts_emotions_in_positive_tweets.pdf'), dpi=300)
    
    
    # Plot emotion counts on a weekly basis
    weekly_counts_emotions_df = positive_tweets_df.groupby(pd.Grouper(freq='W'))[emotion_labels].sum()
    weekly_counts_emotions_df.index = weekly_counts_emotions_df.index.date
    weekly_counts_emotions_df.to_csv(os.path.join(individual_dir, 'weekly_counts_emotions_positive_tweets.csv'))

    fig_emotions2, ax_emotions2 = plt.subplots(nrows=1, ncols=1, figsize=(40,40))
    weekly_counts_emotions_df[emotion_labels].plot.bar(ax=ax_emotions2, stacked=True, colormap=cmap, edgecolor='black')
    ax_emotions2.legend(title='Emotion', loc='upper left', fontsize=fsize1)
    ax_emotions2.set_xlabel('Time', fontsize=fsize1)
    ax_emotions2.set_ylabel('Count', fontsize=fsize1)
    ax_emotions2.tick_params(axis='x', labelsize=fsize2)
    ax_emotions2.tick_params(axis='y', labelsize=fsize2)
    fig_emotions2 = ax_emotions2.get_figure()
    fig_emotions2.savefig(os.path.join(individual_dir, 'weekly_counts_emotions_in_positive_tweets.png'), dpi=300)


if __name__ == '__main__':
    main()
