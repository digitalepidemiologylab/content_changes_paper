umport os
import sys
import glob
import pandas as pd
import multiprocessing

from tqdm import tqdm
import datetime

from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='ekphrasis')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# Disable print
def disable_print():
    sys.stdout = open(os.devnull, 'w')

# Enable print
def enable_print():
    sys.stdout = sys.__stdout__


def timeline_preprocessing(f_name):
    def _preprocess_fn(original_text, preprocessor):
        disable_print()
        preprocessed_text = preprocessor.pre_process_doc(str(original_text))
        preprocessed_text = ' '.join(preprocessed_text)
        enable_print()
        return preprocessed_text
    try:
        df = pd.read_csv(f_name, sep='\t')
    except pd.errors.ParserError:
        logger.error(f'Malformed input file: {f_name}')
    
    disable_print()
    preprocessor = TextPreProcessor(
        omit=['url', 'email', 'user'],
        normalize=['url', 'email', 'user'],
        segmenter="twitter_2018",
        corrector="twitter_2018",
        unpack_hashtags=True,
        unpack_contractions=False,
        tokenizer=SocialTokenizer(lowercase=True).tokenize)
    
    df['preprocessed_text'] = df.Tweet.apply(lambda s: _preprocess_fn(s, preprocessor))
    
    basename = os.path.basename(f_name).split('.')[0]
    df.to_parquet(os.path.join(output_folder, f'{basename}.parquet'), index=False)
    # Free memory
    df = pd.DataFrame()
    enable_print()


def main():
    input_folder = os.path.join('data', 'prepared')
    input_files_list = glob.glob(os.path.join(input_folder, '*.txt'))
    
    num_worker_processes = 16
    pool = multiprocessing.Pool(num_worker_processes)
    logger.info('Start preprocessing tweets...')
    num_input_files = len(input_files_list)
    s_time = datetime.datetime.now()
    pbar = tqdm(total=num_input_files)
    def update(*a):
        pbar.update()
    for input_file in input_files_list:
        pool.apply_async(timeline_preprocessing, args=(input_file,), callback=update)
    pool.close()
    pool.join()
    e_time = datetime.datetime.now()
    process_time = (e_time - s_time).seconds / 60
    logger.info(f'Preprocessed tweets from {num_input_files} files in {process_time:.2f} min')


if __name__ == '__main__':
    output_folder = os.path.join('data', f'ekphrasis_preprocessed')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    main()
