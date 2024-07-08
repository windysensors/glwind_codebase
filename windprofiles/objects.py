### booms.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: Boom object class 

from units import Quantity
import pandas as pd
import atmo_calc

class Boom:
    def __init__(dataframe, height, column_info):
        pass

def boom_from_csv(datafile: str, height: float, column_info: dict[str,tuple[str,str]]) -> Boom:
    df = pd.from_csv(datafile)
    return Boom(df, height, column_info)

def save_boom(filename) -> None:
    pass

def load_boom(filename) -> Boom:
    pass

class MetTower:
    def __init__(booms: list[Boom], lat, long, terrain_class = atmo_calc.cedar_rapids_terrain_class):
        self.booms = booms # list of booms
        self.lat = lat # latitude
        self.long = long # longitude
        self.heights = list(booms.keys()) # heights at which data is available
        self.terrain_class = terrain_class # terrain classification function

def load_tower(filename) -> MetTower:
    pass

