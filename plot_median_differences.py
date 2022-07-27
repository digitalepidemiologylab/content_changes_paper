import os
import pandas as pd
from matplotlib import pyplot as plt


def generate_plots():
    emotion_labels = ["anger", "anticipation", "disgust", "fear", "joy",
                    "love", "neutral", "optimism", "pessimism", 
                    "sadness", "surprise", "trust"]
    topic_labels = ["is_Business & Industry", "is_Computers & Internet & Electronics", 
            "is_Education & Reference", "is_Entertainment & Music",
            "is_Health", "is_other", "is_Politics & Government & Law",
            "is_Science & Mathematics",  "is_Society & Culture", "is_Sport"]
    URL_labels = ['U_cat1_Adult', 'U_cat1_Arts_and_Entertainment', 'U_cat1_Autos_and_Vehicles', 'U_cat1_Beauty_and_Fitness', 'U_cat1_Books_and_Literature', 'U_cat1_Business_and_Industry', 'U_cat1_Career_and_Education', 
            'U_cat1_Computer_and_Electronics', 'U_cat1_Finance', 'U_cat1_Food_and_Drink', 'U_cat1_Gambling', 
            'U_cat1_Games', 'U_cat1_Health', 'U_cat1_Home_and_Garden', 'U_cat1_Internet_and_Telecom',
            'U_cat1_Law_and_Government', 'U_cat1_News_and_Media', 'U_cat1_People_and_Society', 
            'U_cat1_Pets_and_Animals', 'U_cat1_Recreation_and_Hobbies', 'U_cat1_Reference', 'U_cat1_Science', 
            'U_cat1_Shopping', 'U_cat1_Sports', 'U_cat1_Travel']
    symptom_labels = ["Chest Pain", "Coughing", "Dizziness", "Dyspnea", 
                    "Fatigue", "Fever", "Headache", "Malaise", "Myalgia",
                    "Nausea", "Sore Throat", "Vomiting"]
    
    
    run_folder = os.path.join('results_statistical_analysis', 'run', 'collective_tests')
    df = pd.read_csv(os.path.join(run_folder, 'wilcoxon_results_all_weeks_adjusted.xlsx'))

    # Emotions
    emotions_df = df.loc[df.feature.isin(emotion_labels),:].copy()
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    ax.barh(emotions_df.feature, emotions_df.pseudomedian, xerr=[emotions_df.pseudomedian-emotions_df.lb_ci, emotions_df.ub_ci-emotions_df.pseudomedian], color=['dodgerblue']*len(emotions_df), edgecolor='black', capsize=3)
    ax.axvline(0, linestyle='--')
    ax.invert_yaxis()
    ax.set_xlabel('Median difference (post-pre)')
    ax.set_ylabel('Emotion')
    fig.tight_layout()
    fig.savefig(os.path.join(run_folder, 'emotions_median_differences.pdf'))

    # Topics
    topics_df = df.loc[df.feature.isin(topic_labels), :].copy()
    topics_df['feature'] = topics_df.feature.str.replace('is_', '')
    topics_df['feature'] = topics_df.feature.str.replace('other', 'Other')
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    ax.barh(topics_df.feature, topics_df.pseudomedian, xerr=[topics_df.pseudomedian-topics_df.lb_ci, topics_df.ub_ci-topics_df.pseudomedian], color=['dodgerblue']*len(topics_df), edgecolor='black', capsize=3)
    ax.axvline(0, linestyle='--')
    ax.invert_yaxis()
    ax.set_xlabel('Median difference (post-pre)')
    ax.set_ylabel('Topic')
    fig.tight_layout()
    fig.savefig(os.path.join(run_folder, 'topics_median_differences.pdf'))

    # URL categories
    url_df = df.loc[df.feature.isin(URL_labels), :].copy()
    url_df['feature'] = url_df.feature.str.replace('U_cat1_', '')
    url_df['feature'] = url_df.feature.str.replace('_', ' ')
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    ax.barh(url_df.feature, url_df.pseudomedian, xerr=[url_df.pseudomedian-url_df.lb_ci, url_df.ub_ci-url_df.pseudomedian], color=['dodgerblue']*len(url_df), edgecolor='black', capsize=3)
    ax.axvline(0, linestyle='--')
    ax.invert_yaxis()
    ax.set_xlabel('Median difference (post-pre)')
    ax.set_ylabel('URL category')
    fig.tight_layout()
    fig.savefig(os.path.join(run_folder, 'URL_median_differences.pdf'))

    # Symptoms
    symptoms_df = df.loc[df.feature.isin(symptom_labels), :].copy()
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300)
    ax.barh(symptoms_df.feature, symptoms_df.pseudomedian, xerr=[symptoms_df.pseudomedian-symptoms_df.lb_ci, symptoms_df.ub_ci-symptoms_df.pseudomedian], color=['dodgerblue']*len(symptoms_df), edgecolor='black', capsize=3)
    ax.axvline(0, linestyle='--')
    ax.invert_yaxis()
    ax.set_xlabel('Median difference (post-pre)')
    ax.set_ylabel('Symptom')
    fig.tight_layout()
    fig.savefig(os.path.join(run_folder, 'symptoms_median_differences.pdf'))

if __name__ == '__main__':
    generate_plots()
