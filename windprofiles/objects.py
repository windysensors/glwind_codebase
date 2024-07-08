### booms.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: Boom and MetTower object classes, as well as functions for file handling

from units import Quantity
import pandas as pd
import atmo_calc
import traceback

class Boom:
    def __init__(dataframe: pd.DataFrame, height: float|Quantity, column_info: dict[str,tuple[str,str]]):
        self.height = height
        self.data = dataframe

    def save(self, filename: str) -> None:
        pass

    @staticmethod
    def generate(filename: str, height: float|Quantity, column_info: dict[str,tuple[str,str]]) -> Boom:
        # generate boom given a CSV file of raw data, boom height, and column info (formatting?)
        print("Hello!")
        df = pd.from_csv(filename)
        return Boom(df, height, column_info)
    
    @staticmethod
    def load(filename: str) -> Boom:
        # g
        pass
    


class MetTower:
    def __init__(name: str, booms: list[Boom], lat, long, terrain_class = atmo_calc.cedar_rapids_terrain_class):
        self.name = name
        self.booms = booms # list of booms
        self.lat = lat # latitude
        self.long = long # longitude
        self.heights = list(booms.keys()) # heights at which data is available
        self.terrain_class = terrain_class # terrain classification function
        
    def save(self, filename: str):
        try:
            pass
        except:
            print("Failed to save tower")
            traceback.print_exc()
    @staticmethod
    def load(filename: str) -> MetTower:
        pass


# note to self: use .zst compression
