from preprocess.utils.helpers import get_all_data
from utils.helpers import cached
from analysis_helpers.geo_helpers import load_map_data
import os
import pandas as pd
import re
import geopandas as gpd
import shapely.geometry
import numpy as np
import multiprocessing
import joblib
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s') 
log = logging.getLogger(__name__)


@cached('time_helpers_load_sentiment_data')
def load_sentiment_data(model, question_tag='sentiment', label_to_int={'positive': 1, 'neutral': 0, 'negative': -1}, query=None, exclude_neutral=False, contains_keywords=True, exclude_retweets=False,
        exclude_extracted=False, limit_by_countries=(), lang=None, geo_enrichment_type=None, nrows=None, no_parallel=False, __cached=True):
    log.info('Load data...')
    if query is not None:
        extra_cols = ['text']
    limit_by_countries = list(limit_by_countries)
    df = load_all_data(include_flags=False, nrows=nrows, geo_enrichment_type=geo_enrichment_type, __cached=__cached)
    if contains_keywords:
        log.info('Filter by contains keyword...')
        df = df[df.contains_keywords]
    if exclude_retweets:
        log.info('Exclude retweets...')
        df = df[~df.is_retweet]
    if exclude_extracted:
        log.info('Exclude extracted...')
        df = df[~df.extracted_quoted_tweet]
    if lang is not None:
        df = df[df.lang == lang]
    if query is not None:
        log.info(f'Filter by query {query}...')
        df = df[df.text.str.contains(query, flags=re.IGNORECASE)]
    if len(limit_by_countries) > 0:
        log.info(f'Limit by countries {limit_by_countries}...')
        df.dropna(subset=['longitude_enriched', 'latitude_enriched'], inplace=True)
        map_data = load_map_data(level='country', limit_by_countries=limit_by_countries)
        if no_parallel:
            num_cores = 1
        else:
            num_cores = max(multiprocessing.cpu_count() - 1, 1)
        parallel = joblib.Parallel(n_jobs=num_cores)
        sjoin_delayed = joblib.delayed(sjoin_coords_with_country)
        num_splits = min(max(len(df) // 1000, 1), len(df))
        log.info(f'Running spatial join on {limit_by_countries}. using {num_cores} cores to run {num_splits} jobs...')
        df = np.array_split(df, num_splits)
        df = parallel((sjoin_delayed(batch, map_data) for batch in tqdm(df)))
        log.info('Merging results...')
        df = pd.concat(df, axis=0)
        df.index.name = 'created_at'
    if len(df) == 0:
        return df
    # convert to index
    model_col = 'label_{}'.format(model)
    df.rename(columns={model_col: question_tag}, inplace=True)
    if label_to_int is not None:
        df[question_tag] = df[question_tag].apply(lambda s: label_to_int[s])
    if exclude_neutral:
        df = df[df[question_tag] != 0]
    return df[['id', question_tag]]


def sjoin_coords_with_country(df, map_data):
    df['geometry'] = [shapely.geometry.Point(lon, lat) for lon, lat in zip(df.longitude_enriched, df.latitude_enriched)]
    df = gpd.GeoDataFrame(df)
    df.crs = {'init' :'epsg:4326'}
    df = gpd.sjoin(df, map_data, op='within')
    return df

# @cached('get_all_data')
def load_all_data(include_flags=False, nrows=None, geo_enrichment_type=None, __cached=True):
    """Simple wrapper in order to cache results"""
    return get_all_data(include_flags=False, nrows=nrows, geo_enrichment_type=geo_enrichment_type)

