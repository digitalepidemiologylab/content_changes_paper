import os
# from utils.helpers import get_data_folder, get_cache_path, cached
from geo.shapely_helper import ShapelyHelper
import pandas as pd
import numpy as np
import geopandas as gpd
import logging

logger = logging.getLogger(__name__)

# def load_raw_data():
#     df = get_all_data(include_all_data=True, include_flags=False)
#     df = df[df.contains_keywords]
#     df = df[df.has_coordinates | df.has_place_bounding_box]
#     cols = ['id',
#             'text',
#             'place.bounding_box',
#             'place.bounding_box.centroid',
#             'place.bounding_box.area',
#             'place.country_code',
#             'place.full_name',
#             'place.place_type',
#             'latitude',
#             'longitude']
#     # add prediction columns
#     cols += [c for c in df.columns if 'label_' in c or 'probability_' in c]
#     return df[cols]

# def load_data():
#     cache_path = get_cache_path('geo_sentiment_data.csv')
#     if not os.path.isfile(cache_path):
#         print('Load data...')
#         df = load_raw_data()
#         print('Loaded {:,} tweets from raw data...'.format(len(df)))
#         df.to_csv(cache_path)
#     print('Reading cached data...')
#     df = pd.read_csv(cache_path)
#     return pd.read_csv(cache_path)

# def prepare_data(df, label_col='label_fasttext_sentiment_v2'):
#     import iso3166
#     def convert_country_code(s):
#         if isinstance(s, str):
#             return iso3166.countries_by_alpha2[s].alpha3
#         else:
#             return np.nan
#     shapely_helper = ShapelyHelper()
#     print('Prepare data...')
#     map_sentiment = {'positive': 1, 'neutral': 0, 'negative': -1}
#     df.dropna(subset=[label_col], inplace=True)
#     df['sent_index'] = df[label_col].apply(lambda s: map_sentiment[s])
#     # convert to iso_3 country code
#     df['iso_a3'] = df['place.country_code'].apply(convert_country_code)
#     # convert geo data
#     df['coordinates'] = df[['longitude', 'latitude']].apply(shapely_helper.convert_to_coordinate, axis=1)
#     df['place.bounding_box'] = df['place.bounding_box'].apply(shapely_helper.convert_to_polygon)
#     df['place.bounding_box.centroid'] = df['place.bounding_box.centroid'].apply(shapely_helper.convert_to_coordinate_from_list)
#     return df

def load_map_data(level='country', limit_by_countries=None):
    # load country/state level borders
    # Download: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    geo_data_folder = os.path.join(get_data_folder(),  'geodata')
    if level == 'country':
        f_path_countries = os.path.join(geo_data_folder, 'natural_earth_vector', '10m_cultural', 'ne_10m_admin_0_countries.shp') # country borders
        df = gpd.read_file(f_path_countries, encoding='utf-8')
        df.crs = {'init' :'epsg:4326'}
    elif level == 'state':
        f_path_states = os.path.join(geo_data_folder, 'natural_earth_vector', '10m_cultural', 'ne_10m_admin_1_states_provinces.shp') # state/province borders
        df = gpd.read_file(f_path_states, encoding='utf-8')
        df.crs = {'init' :'epsg:4326'}
    elif level == 'county':
        f_path_county = os.path.join(geo_data_folder, 'diva_gis', 'BRA_adm3.shx') # county borders of Brazil
        f_path_county_info = os.path.join(geo_data_folder, 'diva_gis', 'BRA_adm3.csv') # county info
        df = gpd.read_file(f_path_county, encoding='utf-8')
        county_info = pd.read_csv(f_path_county_info)
        df = pd.concat([df, county_info], axis=1)
        df = gpd.GeoDataFrame(df, geometry='geometry')
        df.crs = {'init' :'epsg:4326'}
    else:
        raise ValueError('Unknown level')
    if limit_by_countries is not None and level in ['country', 'state']:
        if not isinstance(limit_by_countries, list):
            raise ValueError('limit_by_countries should be of type list')
        if 'iso_a2' in df:
            df = df[df.iso_a2.isin(limit_by_countries)]
        elif 'ISO_A2' in df:
            df = df[df.ISO_A2.isin(limit_by_countries)]
        else:
            raise Exception('Could not filter by country because map data contains no ISO admin 2 column.')
    return df

def load_population_data(level='country', limit_by_countries=None):
    logger.info('Loading population data...')
    if level == 'country':
        # Source: https://www.prb.org/international/indicator/population/map/country
        f_path = os.path.join(get_data_folder(),  'geodata',  'pop_density', 'population_by_country.csv')
        df = pd.read_csv(f_path, usecols=['FIPS', 'Name', 'Data'])
        df.columns = ['iso_a2', 'country', 'population']
        countries = load_map_data(level='country', limit_by_countries=limit_by_countries)
        df = pd.merge(df, countries, left_on='iso_a2', right_on='ISO_A2', how='inner')
    elif level == 'state':
        # Source: https://ibge.gov.br/
        f_path = os.path.join(get_data_folder(),  'geodata',  'pop_density', 'population_by_state.csv')
        df = pd.read_csv(f_path, usecols=['State', 'Population (2014)[2]'])
        df.columns = ['state', 'population']
        df.population = df.population.str.split(',').str.join('').astype(int)
        states = load_map_data(level='state', limit_by_countries=limit_by_countries)
        states = states[['iso_a2', 'name', 'geometry']]
        df = pd.merge(df, states, left_on='state', right_on='name', how='inner')
    elif level == 'county':
        raise NotImplementedError
    else:
        raise ValueError('Unknown level')
    df = gpd.GeoDataFrame(df, geometry='geometry')
    df.crs = {'init' :'epsg:4326'}
    return df

# @cached('geo_helpers_load_sentiment_data')
# def load_sentiment_data(model, level='state', limit_by_countries=None):
#     print('Loading sentiment data...')
#     df = load_data()
#     df = prepare_data(df, label_col='label_{}'.format(model))
#     df.rename(columns={'coordinates': 'geometry'}, inplace=True)
#     df.geometry.fillna(df['place.bounding_box.centroid'], inplace=True)
#     df = gpd.GeoDataFrame(df)
#     df.crs = {'init' :'epsg:4326'}
#     print ('Join sentiment data with world map on {} level...'.format(level))
#     map_data = load_map_data(level='country' if level == 'raw' else level, limit_by_countries=limit_by_countries)
#     df = gpd.sjoin(df, map_data, op='within')
#     if level == 'raw':
#         df['x'] = df.geometry.apply(lambda s: s.x)
#         df['y'] = df.geometry.apply(lambda s: s.y)
#         return df[['x', 'y', 'geometry', 'sent_index']]
#     else:
#         print ('Compute mean sentiment by {}...'.format(level))
#         map_data['sentiment'] = np.nan
#         map_data['sentiment_count'] = 0
#         if level == 'state':
#             groupby_col = 'adm1_code'
#         elif level == 'county':
#             groupby_col = 'NAME_3'
#         else:
#             ValueError('Invalid level code')
#         for area_code, g in df.groupby(groupby_col):
#             map_data.loc[map_data[groupby_col] == area_code, 'sentiment'] = g.sent_index.mean()
#             map_data.loc[map_data[groupby_col] == area_code, 'sentiment_count'] = g.sent_index.count()
#         return map_data
