import re
import sys
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import iso3166
from utils.analysis_helpers.geo.shapely_helper import ShapelyHelper

# Helpers in the current directory
from utils.cache_helpers import cached, cached_parquet_GeoDataFrame
from utils.map_helpers import load_map_data

import multiprocessing
import joblib
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s') 
log = logging.getLogger(__name__)


def load_data_with_predicted_labels(usecols=None, extra_cols=None):
    # Load data
    if usecols is None:
        # Default columns
        usecols = ['id', 'text', 'label', 'created_at', 'lang', 'contains_keywords', 'token_count', 'is_retweet', 'retweeted_status_id', 'longitude', 'latitude', 'country_code', 'geoname_id', 'location_type', 'geo_type', 'region', 'subregion']
        if extra_cols is not None:
            for ec in extra_cols:
                usecols.append(ec)
    df = pd.read_csv(os.path.join('data', 'predict_data', 'precovid_paho_290621_sentiment', 'all_features_predictions_2021-07-13_09-40-16_018009', 'tweets_with_predicted_labels_2013-01-01_to_2019-12-31.csv'), usecols=usecols)
    # Convert to category
    for col in ['country_code', 'region', 'subregion', 'geo_type', 'lang']:
        if col in df:
            df[col] = df[col].astype('category')
    # Convert the 'created_at' column to datetime format
    df['created_at'] = pd.to_datetime(df['created_at'])
    # Change index column
    df = df.set_index('created_at')
    return df

# @cached_parquet_GeoDataFrame('prepare_data.parquet')
def prepare_data(df):
    shapely_helper = ShapelyHelper()
    # Convert geo data
    df['coordinates'] = df[['longitude', 'latitude']].apply(shapely_helper.convert_to_coordinate, axis=1)
    df.rename(columns={'coordinates': 'geometry'}, inplace=True)
    # df is a DataFrame and gdf is the GeoDataFrame object obtained from df
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    # Coordinate reference system: latitude/longitude
    gdf.crs = 'epsg:4326'
    log.info('Data moved into a GeoDataFrame...')
    return gdf

def convert_country_code(s):
    if isinstance(s, str):
        return iso3166.countries_by_alpha2[s].alpha3
    else:
        return np.nan

# @cached('time_helpers_load_sentiment_data')
def load_sentiment_data(question_tag='sentiment', label_to_int={'positive': 1, 'neutral': 0, 'negative': -1}, query=None, exclude_neutral=False, contains_keywords=True, exclude_retweets=False,
        limit_by_countries=None, lang=None, geo_enrichment_type=None, nrows=None, no_parallel=False):

    log.info('Load data...')
    df = load_data_with_predicted_labels()
    df.reset_index(inplace=True)
    df.index.name = 'number'
    df.reset_index(inplace=True)
    df.set_index('created_at', inplace=True)
    # Columns of the DataFrame df: 'id', 'number', 'text', 'label', 'lang', 'contains_keywords', 'token_count', 'is_retweet', 'retweeted_status_id', 'longitude', 'latitude', 'country_code', 'geoname_id', 'location_type', 'geo_type', 'region', 'subregion'
    # The index is the column 'created_at'
    if contains_keywords:
        log.info('Filter by contains_keywords...')
        df = df[df.contains_keywords]
    if exclude_retweets:
        log.info('Exclude retweets...')
        df = df[~df.is_retweet]
    if lang is not None:
        df = df[df.lang == lang]
    if query is not None:
        log.info(f'Filter by query {query}...')
        df = df[df.text.str.contains(query, flags=re.IGNORECASE)]

    limit_by_countries = list(limit_by_countries)
    if len(limit_by_countries) > 0: 
        log.info('Filter entries with relevant geo_type values and location_type values...')
        # We will not keep tweets for which no geolocation could be inferred (geo_type == 0). Note: whenever geo_type == 0, location_type == None (default value for location_type)
        # We will keep all tweets for which exact coordinates were provided by Twitter (geo_type == 1). Note: whenever geo_type == 1, location_type == None (default value for location_type)
        # There are 5 possible values of location_type if geo_type == 2: ['country', 'city', 'admin', 'poi', 'neighborhood']. Note: 'poi' stands for 'place of interest'
        # When geo_type == 2, we will only keep the following entries (4 values): ['city', 'admin', 'poi', 'neighborhood']
        # There are 12 possible values of location_type if geo_type == 3: ['continent', 'region', 'country', 'city', 'place', 'admin1', 'admin2', 'admin3', 'admin4', 'admin5', 'admin6', 'admin_other']
        # When geo_type == 3, we will only keep the following entries (9 values): ['city', 'place', 'admin1', 'admin2', 'admin3', 'admin4', 'admin5', 'admin6', 'admin_other']
        df = df[~((df.geo_type == 0) | ((df.geo_type == 2) & (df.location_type.isin(['country']))) | ((df.geo_type == 3) & (df.location_type.isin(['country','continent','region']))))]
        # The new assignment of df is a GeoDataFrame
        # df = prepare_data(df, __cached=False)
        df = prepare_data(df)
        
        log.info(f'Limit by countries {limit_by_countries}...')
        # map_data_df is a GeoDataFrame
        map_data_df = load_map_data('state', limit_by_countries=limit_by_countries)
        # Derive the three-letter country codes (iso_a3/ISO_A3) from the corresponding two-letter country codes (iso_a2/ISO_A2) 
        if 'iso_a2' in map_data_df.columns:
            map_data_df['iso_a3'] = map_data_df.iso_a2.apply(convert_country_code)
        if 'ISO' in map_data_df.columns:
            map_data_df = map_data_df.rename(columns={'ISO': 'iso_a3'})
        
        # Spatial join between geo_sentiment_df and map_data_df
        log.info('Join sentiment data with world map...')
        df = gpd.sjoin(df, map_data_df, op='within')
        df.index.name = 'created_at'
        # Sort object by labels
        df.sort_index(inplace=True)
    if len(df) == 0:
        return df
    # Rename column
    df.rename(columns={'label': question_tag}, inplace=True)
    # Map sentiment labels to numerical values
    if label_to_int is not None:
        df[question_tag] = df[question_tag].apply(lambda s: label_to_int[s])
        # Remove 'nan' entries
        df = df.dropna(subset=[question_tag])
        df = df[df[question_tag] != 'nan']
    if exclude_neutral:
        df = df[df[question_tag] != 0]
    return df[['id', 'number', question_tag]]

def load_sentiment_data_bis(question_tag='sentiment', label_to_int={'positive': 1, 'neutral': 0, 'negative': -1}, query=None, exclude_neutral=False, contains_keywords=True, exclude_retweets=False,
        limit_by_countries=None, lang=None, geo_enrichment_type=None, nrows=None, no_parallel=False):

    log.info('Load data...')
    df = load_data_with_predicted_labels()
    df.reset_index(inplace=True)
    df.index.name = 'number'
    df.reset_index(inplace=True)
    df.set_index('created_at', inplace=True)
    # Columns of the DataFrame df: 'id', 'number', 'text', 'label', 'lang', 'contains_keywords', 'token_count', 'is_retweet', 'retweeted_status_id', 'longitude', 'latitude', 'country_code', 'geoname_id', 'location_type', 'geo_type', 'region', 'subregion'
    # The index is the column 'created_at'
    if contains_keywords:
        log.info('Filter by contains_keywords...')
        df = df[df.contains_keywords]
    if exclude_retweets:
        log.info('Exclude retweets...')
        df = df[~df.is_retweet]
    if lang is not None:
        df = df[df.lang == lang]
    if query is not None:
        log.info(f'Filter by query {query}...')
        df = df[df.text.str.contains(query, flags=re.IGNORECASE)]

    limit_by_countries = list(limit_by_countries)
    if len(limit_by_countries) > 0: 
        log.info('Filter entries with relevant geo_type values and location_type values...')
        # We will not keep tweets for which no geolocation could be inferred (geo_type == 0). Note: whenever geo_type == 0, location_type == None (default value for location_type)
        # We will keep all tweets for which exact coordinates were provided by Twitter (geo_type == 1). Note: whenever geo_type == 1, location_type == None (default value for location_type)
        # There are 5 possible values of location_type if geo_type == 2: ['country', 'city', 'admin', 'poi', 'neighborhood']. Note: 'poi' stands for 'place of interest'
        # We will keep all tweets with geo_type = 2
        # There are 12 possible values of location_type if geo_type == 3: ['continent', 'region', 'country', 'city', 'place', 'admin1', 'admin2', 'admin3', 'admin4', 'admin5', 'admin6', 'admin_other']
        # When geo_type == 3, we will only keep the following entries (10 values): ['country', 'city', 'place', 'admin1', 'admin2', 'admin3', 'admin4', 'admin5', 'admin6', 'admin_other']
        df = df[~((df.geo_type == 0) | ((df.geo_type == 3) & (df.location_type.isin(['continent','region']))))]
        # The new assignment of df is a GeoDataFrame
        # df = prepare_data(df, __cached=False)
        df = prepare_data(df)
        
        log.info(f'Limit by countries {limit_by_countries}...')
        # map_data_df is a GeoDataFrame
        map_data_df = load_map_data('state', limit_by_countries=limit_by_countries)
        # Derive the three-letter country codes (iso_a3/ISO_A3) from the corresponding two-letter country codes (iso_a2/ISO_A2) 
        if 'iso_a2' in map_data_df.columns:
            map_data_df['iso_a3'] = map_data_df.iso_a2.apply(convert_country_code)
        if 'ISO' in map_data_df.columns:
            map_data_df = map_data_df.rename(columns={'ISO': 'iso_a3'})
        
        # Spatial join between geo_sentiment_df and map_data_df
        log.info('Join sentiment data with world map...')
        df = gpd.sjoin(df, map_data_df, op='within')
        df.index.name = 'created_at'
        # Sort object by labels
        df.sort_index(inplace=True)
    if len(df) == 0:
        return df
    # Rename column
    df.rename(columns={'label': question_tag}, inplace=True)
    # Map sentiment labels to numerical values
    if label_to_int is not None:
        df[question_tag] = df[question_tag].apply(lambda s: label_to_int[s])
        # Remove 'nan' entries
        df = df.dropna(subset=[question_tag])
        df = df[df[question_tag] != 'nan']
    if exclude_neutral:
        df = df[df[question_tag] != 0]
    return df[['id', 'number', question_tag]]
