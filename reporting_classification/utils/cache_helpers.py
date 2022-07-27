import os
import pandas as pd
import geopandas as gpd
import pickle
from functools import wraps
import shelve

def get_data_folder():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    f_path = os.path.join(current_folder, '..', 'data')
    if os.path.isdir(f_path):
        return os.path.abspath(f_path)
    else:
        raise FileNotFoundError('Folder {0} could not be found.'.format(folder_path))

def cache_folder(use_data_folder=False, subfolder=None):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if use_data_folder:
        f_path = os.path.join(current_folder, '..', 'data')
    else:
        if subfolder is None:
            f_path = os.path.join(current_folder, '..', 'data', 'cache')
        else:
            f_path = os.path.join(current_folder, '..', 'data', 'cache', subfolder)
    if not os.path.isdir(f_path):
        os.makedirs(f_path)
    return os.path.abspath(f_path)

def get_cache_path(f_name, use_data_folder=False, subfolder=None):
    f_path = os.path.join(cache_folder(subfolder=subfolder, use_data_folder=use_data_folder), f_name)
    return os.path.abspath(f_path)

def find_project_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])

def cached(f_name):
    """Uses a shelve to pickle return values of function calls"""
    cache_path = get_cache_path(f_name)
    def cacheondisk(fn):
        db = shelve.open(cache_path)
        @wraps(fn)
        def usingcache(*args, **kwargs):
            __cached = kwargs.pop('__cached', True)
            key = repr((args, kwargs))
            if not __cached or key not in db:
                ret = db[key] = fn(*args, **kwargs)
            else:
                print(f'Using cache')
                ret = db[key]
            return ret
        return usingcache
        db.close()
    return cacheondisk

def cached_parquet(f_name):
    """Uses a parquet to cache return values of function calls"""
    if not f_name.endswith('.parquet'):
        f_name += '.parquet'
    cache_path = get_cache_path(f_name, subfolder='cached_parquet')
    def cacheondisk(fn):
        @wraps(fn)
        def usingcache(*args, **kwargs):
            __cached = kwargs.pop('__cached', True)
            key = repr((args, kwargs))
            if not __cached or not os.path.isfile(cache_path):
                ret = fn(*args, **kwargs)
                ret.to_parquet(cache_path)
            else:
                print(f'Using cache')
                ret = pd.read_parquet(cache_path)
            return ret
        return usingcache
    return cacheondisk

def cached_parquet_GeoDataFrame(f_name):
    """Uses a parquet to cache return values of function calls"""
    if not f_name.endswith('.parquet'):
        f_name += '.parquet'
    cache_path = get_cache_path(f_name, subfolder='cached_parquet')
    def cacheondisk(fn):
        @wraps(fn)
        def usingcache(*args, **kwargs):
            __cached = kwargs.pop('__cached', True)
            key = repr((args, kwargs))
            if not __cached or not os.path.isfile(cache_path):
                ret = fn(*args, **kwargs)
                ret.to_parquet(cache_path)
            else:
                print(f'Using cache')
                ret = gpd.read_parquet(cache_path)
            return ret
        return usingcache
    return cacheondisk
