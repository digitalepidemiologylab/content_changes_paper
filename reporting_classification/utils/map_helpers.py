import os
import json
import pandas as pd
import geopandas as gpd
from utils.cache_helpers import get_data_folder, cached_parquet_GeoDataFrame
import wget
import zipfile
import logging
logger = logging.getLogger(__name__)

# First-level administrative divisions of Brazil are called 'unidades federativas do Brasil' (27 federative units of Brazil) and are represented by 26 'estados' (states) and the 'Distrito Federal' (Federal District).
# Second-level administrative divisions of Brazil are called 'municípios' (municipalities). These are administrative divisions of Brazilian states; the Federal District cannot be divided into municipalities, which is why its territory is composed of several administrative regions ('regiões administrativas do Distrito Federal').
# Third-level administrative divisions of Brazil are called 'distritos' (districts). These are administrative divisions of municipalities. Almost all municipalities are divided into 'bairros' (neighbourhoods).
@cached_parquet_GeoDataFrame('map_data_states.parquet')
def load_map_data(agg_level, limit_by_countries=None):
    # Download: https://www.naturalearthdata.com/downloads/10m-cultural-vectors/
    geo_data_folder = os.path.join(get_data_folder(),  'geodata')
    all_countries_fpath = os.path.join(geo_data_folder, 'ISO-3166-Countries-with-Regional-Codes', 'all', 'all.json')

    with open(all_countries_fpath, 'r') as f:
        # List of dictionaries containing information about ISO-3166 Countries and Dependent Territories with UN Regional Codes
        all_countries_list = json.load(f)

    # Use the content of this list with a DataFrame
    all_countries_df = pd.DataFrame(all_countries_list)
    # Select the relevant columns (country names and associated three-letter country codes)
    name_alpha3_df = all_countries_df[['name', 'alpha-3']].copy()

    # Create a dictionary with the country names as keys and the ISO 3166-1 alpha-3 codes as values
    alpha3_name_dict = {}
    for i in range(len(name_alpha3_df)):
        alpha3_name_dict[name_alpha3_df.loc[i, 'alpha-3']] = name_alpha3_df.loc[i, 'name']

    if agg_level == 'country':
        logger.info('Looking for data (from https://www.naturalearthdata.com/) at the international level...')
        f_path_countries = os.path.join(geo_data_folder, '10m_cultural', 'ne_10m_admin_0_countries.shp') # country borders
        df = gpd.read_file(f_path_countries, encoding='utf-8')
        df.crs = 'epsg:4326'
    elif agg_level == 'state':
        logger.info('Looking for data (from https://www.naturalearthdata.com/) at the first administrative level of the countries...')
        f_path_states = os.path.join(geo_data_folder, '10m_cultural', 'ne_10m_admin_1_states_provinces.shp') # state/province borders
        df = gpd.read_file(f_path_states, encoding='utf-8')
        df.crs = 'epsg:4326'
    elif agg_level == 'county':
        logger.info('Looking for data (from https://www.diva-gis.org/) at the second administrative level of the countries...')
        if limit_by_countries is None or not isinstance(limit_by_countries, list) or len(limit_by_countries) == 0:
            raise ValueError('Please give a non-empty list of country names for which you would like to have maps at the county level')
        else:
            country_names_series = pd.Series(limit_by_countries)
            country_names_keys = list(alpha3_name_dict.keys())
            if not all(country_names_series.isin(country_names_keys)):
                raise ValueError('Some of the country names present in the list were not found in the dictionary containing the names of all the available countries')
            counties_dict = {}
            for cname in limit_by_countries:
                alpha3_current = cname
                alpha3_adm = alpha3_current + '_adm'
                subfolder_path = os.path.join(geo_data_folder, alpha3_adm)
                download_url_prefix = 'https://biogeo.ucdavis.edu/data/diva/adm/'
                logger.info('Looking for existing geodata about {}...'.format(alpha3_name_dict[cname]))
                if not os.path.isdir(subfolder_path):
                    os.mkdir(subfolder_path)
                    logger.info('No existing geodata about {c_name} found... Downloading data from {download_url_prefix} ...'.format(c_name=c_name, download_url_prefix=download_url_prefix))
                    zipped_adm_file = wget.download(download_url_prefix + alpha3_adm + '.zip')
                    # Move subfolder named alpha3_adm to the correct location, i.e., subfolder_path
                    zipped_adm_fpath = shutil.move(zipped_adm_file, subfolder_path)
                    # Unzip
                    archive = zipfile.ZipFile(zipped_adm_fpath)
                    archive.extractall(path=subfolder_path)
                
                # Identify the name of the file describing the borders of the second-level administrative divisions
                adm2_shp_fpath = os.path.join(subfolder_path, alpha3_adm + '2.shp') 
                # # Identify the name of the file containing information about the second-level administrative divisions
                # adm2_csv_fpath = os.path.join(subfolder_path, alpha3_adm + '2.csv') 
                
                counties_geo_df = gpd.read_file(adm2_shp_fpath, encoding='utf-8')
                # counties_info_df = pd.read_csv(adm2_csv_fpath, usecols=['OBJECTID']) # The only difference between the Shape and CSV files is that the latter contain a column called 'OBJECTID' – which is not useful – instead of the column called 'geometry' in the CSV files, hence adding the original content of the CSV files is optional
                # counties_geo_df = pd.concat([counties_geo_df, counties_info_df], axis=1)
                counties_geo_df = gpd.GeoDataFrame(counties_geo_df, geometry='geometry')
                counties_geo_df.crs = 'epsg:4326'
                counties_dict[alpha3_current] = counties_geo_df

                df = pd.concat(counties_dict.values(), axis=0)
                df = df.reset_index().drop('index', axis=1)
        
    else:
        raise ValueError('Unknown level')
    if limit_by_countries is not None and agg_level in ['country', 'state']:
        if not isinstance(limit_by_countries, list):
            raise TypeError('limit_by_countries should be of type list')
        if 'ISO_A2' in df.columns:
            df = df.rename(columns={'ISO_A2': 'iso_a2'})
        if 'iso_a2' in df.columns:
            df = df[df.iso_a2.isin(limit_by_countries)]
        else:
            raise Exception('Could not filter by country because map data contains no ISO 3166-1 alpha-2 column ("iso_a2"/"ISO_A2"; two-letter country code).')
    return df
