import os
import pandas as pd
import numpy as np
from causalimpact import CausalImpact
from matplotlib import pyplot as plt


def save_ci_figure(ci, input_folder):
    ci.plot(show=False)
    fig = plt.gcf()
    axes_list = fig.axes
    # Add labels on the y-axis
    fsize = 7 # Fontsize
    axes_list[0].set_ylabel('Counts', fontsize=fsize)
    # Pointwise contributions represent the difference between observed counts and counterfactual predictions
    axes_list[1].set_ylabel('Pointwise contributions', fontsize=fsize)
    # Cumulative effect
    axes_list[2].set_ylabel('Sum of the pointwise contributions', fontsize=fsize)

    # Label for the x-axis
    axes_list[2].set_xlabel('Number of weeks elapsed since the first tweet')

    # Add title to the figure
    # fig.suptitle('Causal impact analysis')
    # Create output folder
    output_folder = os.path.join(os.path.dirname(input_folder), 'plots', os.path.basename(input_folder))
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    # Save figure
    fig.savefig(os.path.join(output_folder, 'causal_impact_analysis_plots.pdf'), dpi=300)


def pre_post_analysis_single_user(user_df, input_folder, significance_level=0.05):
    user_id = user_df['user.id'].unique()[0]
    output_df = pd.DataFrame(index=[user_id])
    output_df.index.rename('user.id', inplace=True)

    min_week_user = user_df.rel_effective_week.min()
    max_week_user = user_df.rel_effective_week.max()
    all_weeks = np.arange(min_week_user, max_week_user+1).tolist()
    week0_position = all_weeks.index(0)
    pre_period = [0, week0_position - 1]
    post_period = [week0_position, len(all_weeks)-1]
    
    available_weeks_counts = user_df.groupby('rel_effective_week')['id'].count()
    
    available_weeks = set(available_weeks_counts.index)
    missing_weeks = list(set(all_weeks) - available_weeks)
    missing_weeks_counts = pd.Series([0]* len(missing_weeks))
    missing_weeks_counts.index = list(missing_weeks)
    weekly_rates_user = pd.concat([available_weeks_counts, missing_weeks_counts]).sort_index()
    data = weekly_rates_user.reset_index(drop=True)
    
    # Apply causal impact analysis
    ci = CausalImpact(data, pre_period, post_period)
    p_value = ci.p_value

    save_ci_figure(ci, input_folder)


def main():
    # Path to the file with all tweets from selected users (12121 Twitter accounts)
    # Note that this CSV file does *not* contain the primary positivity tweets
    run_name = 'run'
    input_folder = os.path.join('results_statistical_analysis', run_name) 
    fpath = os.path.join(input_folder, 'individual_tests', 'tweets_selected_users_pre_post_periods.csv')
    
    df_selected_users = pd.read_csv(fpath)
    # Select single user: DEFINE A USER ID HERE (REPLACE VALUE X) 
    # user_id = X 
    user_df = df_selected_users.loc[df_selected_users['user.id']==user_id].copy()

    pre_post_analysis_single_user(user_df, input_folder)


if __name__ == '__main__':
    main()
