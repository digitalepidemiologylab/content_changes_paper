import shapely.geometry
import pandas as pd
import numpy as np
import ast

class ShapelyHelper():
    """Various helpers to transform python from and into shapely objects"""

    def __init__(self):
        pass

    def convert_to_coordinate(self, row):
        """Takes pandas dataframe row with longitude and latitude column and returns shapely Point"""
        if pd.isna(row.longitude) or pd.isna(row.latitude):
            return np.nan
        else:
            return shapely.geometry.Point(row.longitude, row.latitude)

    def convert_to_coordinate_from_list(self, l):
        """Takes stringified python list of longitude/latitude pair and returns shapely Point"""
        if pd.isna(l):
            return np.nan
        else:
            l = ast.literal_eval(l)
            return shapely.geometry.Point(l[0], l[1])

    def convert_to_polygon(self, s):
        """Takes stringified python list of polygon coordinates and returns shapely Polygon"""
        if pd.isna(s):
            return np.nan
        else:
            s = ast.literal_eval(s)
            for i, _s in enumerate(s):
                s[i] = [float(_s[0]), float(_s[1])]
            return shapely.geometry.Polygon(s)
