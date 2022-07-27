import pandas as pd
import matplotlib.pyplot as plt

import os
import glob

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

def main():
    dataset_dir = os.path.join('data', 'predict_data', 'dataset')
    predictions_run = 'predictions'
    folder_path = os.path.join(dataset_dir, predictions_run, 'predictions')
    num_pred_files = len(glob.glob(os.path.join(folder_path, '*.jsonl')))
    basename = 'text_medcat_timelines_'
    prediction_files = [basename + str(i) + '.jsonl' for i in range(1,num_pred_files+1)]

    logger.info(f'Load predictions ({len(prediction_files)} files)...')
    pred_df = pd.DataFrame()
    for fname in prediction_files:
        df = pd.read_json(os.path.join(folder_path,fname), lines=True)
        pred_df = pd.concat([pred_df, df], axis=0)
    # Reset index
    pred_df.reset_index(drop=True, inplace=True)

    original_files_path = os.path.join('data', 'data_for_preprocessing')
    # Load tweets with normalized text
    norm_text_df = pd.read_csv(os.path.join(original_files_path, 'text.csv'), header=None)
    # Add column name
    norm_text_df.columns = ['normalized_text']
    # The text of the tweets in the following file is not normalized (hence it is different from the one in norm_text_df)
    userid_tweetid_df = pd.read_parquet(os.path.join(original_files_path, 'id_text.parquet'), columns = ['user.id', 'id']) 
    # Reset index
    userid_tweetid_df.reset_index(drop=True, inplace=True)

    logger.info('Concatenate DataFrames with the user IDs, tweet IDs, text fields, and predicted labels...')
    # Concatenation
    final_df = pd.concat([userid_tweetid_df, norm_text_df, pred_df], axis=1)
    # Save content of the DataFrame
    final_path = os.path.join('data', 'data_after_postprocessing', predictions_run)
    if not os.path.isdir(final_path):
        os.makedirs(final_path)
    logger.info('Save resulting DataFrame...')
    final_df.to_parquet(os.path.join(final_path, 'ids_text_all_predictions.parquet'))
    logger.info("Save DataFrame with filtered predictions (only 'Self_reports')...")
    filtered_df = final_df.loc[final_df.label == 'Self_reports'].copy()
    sorted_selfreport_values = filtered_df.label_probabilities.apply(lambda x: x['Self_reports']).sort_values(ascending=False)
    plt.hist(sorted_selfreport_values, bins=20, range=(0,1))
    plt.xlabel("'Self_reports' probabilities")

    plt.ylabel("Counts")
    plt.savefig(os.path.join(final_path, 'Hist_probabilities_Self_reports_linscale.png'))

    plt.ylabel("Log Counts")
    plt.yscale('log')
    plt.savefig(os.path.join(final_path, 'Hist_probabilities_Self_reports_logscale.png'))

    filtered_df = filtered_df[filtered_df.label_probabilities.apply(lambda x: x['Self_reports'] >=0.9)]
    filtered_df.to_parquet(os.path.join(final_path, 'ids_text_filtered_predictions.parquet'))

    logger.info('Done!')

if __name__ == '__main__':
    main()
