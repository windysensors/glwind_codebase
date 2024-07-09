### booms.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: Boom and MetTower object classes

from __future__ import annotations
from units import Quantity, defaults
from units import convert_value as cnv
import stat_calc
import pandas as pd
import numpy as np
import atmo_calc
import traceback
import os
from exceptions import *

# helper function for initializing Boom class: renames columns and converts their units
def _nameAndConvert(df, old, dim, hid) -> None:
    oldName, oldUnits = old
    if oldName not in df.columns:
        raise NameNotFound(f"Column '{oldName}' not found in data for boom @{hid}m")
        return
    df.rename(columns={oldName:dim}, inplace=True)
    defaultUnits = defaults[dim]
    if oldUnits != defaultUnits:
        df[dim] = cnv(df[dim], oldUnits, defaultUnits, infer=False, dimension = dim)
    return

# helper function for initializing Boom class: renames time column and converts to pd.DateTime format
def _timeName(df, oldName, hid) -> None:
    if oldName not in df.columns:
        raise NameNotFound(f"Column '{oldName}' not found in data for boom @{hid}m")
        return
    df.rename(columns={oldName:'time'}, inplace=True)
    df['time'] = pd.to_datetime(df['time']) # convert to common pandas DateTime type
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

# Boom object class
# Structure for holding time-series data from a single met tower boom (pd.DataFrame underlying)
# Methods for merging with other Booms as well as basic data handling (outlier removal, resampling)
class Boom:
    def __init__(self, dataframe: pd.DataFrame, height: float|Quantity, *,
                 time: str='TIMESTAMP', ws: tuple[str,str]=None, wd: tuple[str,str]=None,
                 P: tuple[str,str]=None, rh: tuple[str,str]=None, T: tuple[str,str]=None,
                 automatic: bool=False, notes: list[str]=[]):
        df = dataframe.copy()
        # handle height - convert to meters
        if type(height) == Quantity:
            self.height = height.in_units_of('m')
        else:
            self.height = float(height)
        # height id - integer part of height
        self.hid = int(self.height)
        if not automatic:
            # Relabel columns according to format specified in _nameAndConvert, and convert to the common units (see units.defaults)
            _timeName(df, time, self.hid)
            if ws: _nameAndConvert(df, ws, 'ws', self.hid)
            if wd: _nameAndConvert(df, wd, 'wd', self.hid)
            if P: _nameAndConvert(df, P, 'P', self.hid)
            if rh: _nameAndConvert(df, rh, 'rh', self.hid)
            if T: _nameAndConvert(df, T, 'T', self.hid)
        self._data = df
        self.available_data = set(self._data.columns.values.tolist())
        self.notes = list(notes)

    def __add__(self, other: Boom) -> Boom:
        # shorthand for Boom inner merge by overloading operator+
        return self.merge(other)
    
    def add_note(self, note: str, verbose=False) -> None:
        self.notes.append(note)
        if verbose:
            print(f'Appended note to boom @{self.hid}m. {len(self.notes)} notes available.')
    
    def print_notes(self) -> None:
        for i, note in enumerate(self.notes):
            print(f'Note {i+1}:\n\t{note}')

    def print_data(self) -> None:
        print(self._data)

    def copy_data(self) -> pd.DataFrame:
        return self._data.copy()

    def save(self, filename: str) -> None:
        # save boom to file (compressed as .zst)None
        pass

    def merge(self, other: Boom, how: str='inner', keep_notes: bool=True) -> Boom:
        # for merging two booms located at the same height, such as booms 6 and 7 for the Iowa data
        # vector average wind speed and direction, standard average other values which are duplicated
        if self.height != other.height:
            raise IncompatibilityError(f'Could not merge booms @{self.hid}m and {other.hid}m - heights must coincide.')
            return self
        dfM = self._data.merge(other._data, on='time', how=how, suffixes=('_1','_2'))
        dfM.sort_index()
        intersection = self.available_data & other.available_data  # data common to both
        if 'ws' in intersection and 'wd' in intersection:
            # wind vector avg
            selfRad = np.deg2rad(dfM['wd_1'])
            selfU = dfM['ws_1'] * np.sin(selfRad)
            selfV = dfM['ws_1'] * np.cos(selfRad)
            otherRad = np.deg2rad(dfM['wd_2'])
            otherU = dfM['ws_2'] * np.sin(otherRad)
            otherV = dfM['ws_2'] * np.cos(otherRad)
            avgU = (selfU + otherU)/2
            avgV = (selfV + otherV)/2
            del selfRad, otherRad, selfU, otherU, selfV, otherV
            dfM['ws'] = np.sqrt(avgU**2 + avgV**2)
            dfM['wd'] = (np.rad2deg(np.arctan2(avgU, avgV)) + 360) % 360
            dfM.drop(columns=['wd_1','wd_2','ws_1','ws_2'], inplace=True)
        elif 'ws' in intersection:
            # just avg ws
            dfM['ws'] = (dfM['ws_1']+dfM['ws_2'])/2
            dfM.drop(columns=['ws_1','ws_2'], inplace=True)
        elif 'wd' in intersection:
            # just (modulo 360-)avg wd
            dfM['wd'] = stat_calc.angle_average(dfM['wd_1'],dfM['wd_2'])
            dfM.drop(columns=['wd_1','wd_2'], inplace=True)
        for variable in intersection-{'ws','wd'}:
            # average other common variables
            dfM[variable] = (dfM[f'{variable}_2']+dfM[f'{variable}_2'])/2
            dfM.drop(columns=[f'{variable}_1',f'{variable}_2'], inplace=True)
        new_notes = []
        if keep_notes:
            new_notes = self.notes + other.notes
        return Boom(dfM, self.height, automatic=True, notes=new_notes)
            
    def partial_merge(self, other: Boom, which_data: str, how: str='inner', inplace: bool=True, add_note: bool=True) -> None|Boom:
        # for merging a column from another boom onto this boom and treating it as if it were from this height
        # this is what is done with boom 1's pressure data onto boom 2 for the Iowa data
        dfM = self._data.copy().join(other._data[which_data], how=how, on='time')
        dfM.sort_index()
        # INCOMPLETE
        note = f"Data of type '{which_data}' is taken from boom @{other.hid}m"
        if inplace:
            if add_note:
                self.add_note(note)
            self._data = dfM
            self.available_data = set(self._data.columns.values.tolist())
            return
        return Boom(dfM, self.height, automatic=True, notes=self.notes+[note]*add_note)

    def delete_column(self, which_data: str):
        if hasattr(which_data, '__iter__'):
            self._data.drop(columns=list(which_data))
        else:
            self._data.drop(columns=[which_data])

    def remove_outliers(self, n_samples: int=30, sigma: float=5., inplace: bool=True, verbose: bool=False, add_note: bool=True) -> None|Boom:
        # removal of outliers which are more than <sigma> standard deviations from a running mean of <n_samples> consecutive samples
        df = self._data.copy()
        eliminations = 0
        dont_consider = ['wd','time']
        to_consider = [col for col in df.columns.values.tolist() if col not in dont_consider]
        for column in to_consider:
            rolling_mean = df[column].rolling(window=f'{int(n_samples)}T').mean()
            rolling_std = df[column].rolling(window=f'{int(n_samples)}T').std()
            threshold = sigma * rolling_std
            outliers = np.abs(df[column] - rolling_mean) > threshold
            eliminations += df[outliers].shape[0]
            df = df[~outliers]
        if verbose:
            print(f'{eliminations} outliers eliminated ({100*eliminations/(df.shape[0]+eliminations):.4f}%)')
        note = f"Outlier removal performed (n_samples = {n_samples}, sigma = {sigma}), removing {eliminations} values"
        if inplace:
            if add_note:
                self.add_note(note)
            self._data = df
            return eliminations
        return Boom(df,self.height,automatic=True,notes=self.notes+[note]*add_note)

    def resample(self, n_samples: int=10, inplace: bool=True, verbose: bool=False, add_note: bool=True) -> int|Boom:
        # resampling by averaging <n_samples> consecutive samples
        df = self._data.copy()
        dirRad = np.deg2rad(df['wd'])
        df['u'] = df['ws'] * np.sin(dirRad)
        df['v'] = df['ws'] * np.cos(dirRad)
        df_avg = df.resample(f'{int(n_samples)}T').mean()
        before_dropna = len(df_avg)
        df_avg.dropna(axis=0,how='all',inplace=True) # If any row is completely blank for some reason, drop it
        eliminations = before_dropna - (len(df_avg))
        df_avg['ws'] = np.sqrt(df_avg['u']**2+df_avg['v']**2)
        df_avg['wd'] = (np.rad2deg(np.arctan2(df_avg['u'], df_avg['v'])) + 360) % 360
        df_avg.drop(columns=['u','v'],inplace=True)
        note = f'Resampled (n_samples = {n_samples})'
        if verbose:
            if eliminations > 0:
                print(f'{eliminations} blank row(s) removed')
            print(note)
        if inplace:
            self._data = df_avg
            if add_note:
                self.add_note(note)
            return eliminations
        return Boom(df_avg,self.height,automatic=True,notes=self.notes+[note]*add_note)

    def vpt_computable(self) -> bool:
        return {'rh','P','T'}.issubset(self.available_data)

    def compute_vpt(self) -> None:
        # compute virtual potential temperature at each time, creating a new column (in place)
        if not self.vpt_computable():
            raise AvailabilityError(f'Not enough data to compute virtual potential temperature @{self.hid}m')
            return
        self._data['vpt'] = atmo_calc.virtual_potential_temperature(self._data['rh'], self._data['P'], self._data['T'])

    @staticmethod
    def generate(filename: str, height: float|Quantity, *,
                 time: str='TIMESTAMP', ws: tuple[str,str]=None, wd: tuple[str,str]=None,
                 P: tuple[str,str]=None, rh: tuple[str,str]=None, T: tuple[str,str]=None,
                 verbose: bool=False):
        # generate boom given a CSV file of raw data, along with boom height, and column info
        df = pd.read_csv(filename)
        result = Boom(df, height, time=time, ws=ws, wd=wd, P=P, rh=rh, T=T)
        if verbose:
            print(f"Generated boom at height {height} from file '{os.path.abspath(filename)}'.")
        return result
    
    @staticmethod
    def load(filename: str) -> Boom:
        # load boom from file (with .zst compression, with the formatting resulting from Boom.save)
        pass
    
class Sonic:
    def __init__(self):
        pass

    def associate(self, other: Boom):
        pass

class MetTower:
    def __init__(self, name: str, booms: list[Boom], latitude: float, longitude: float, terrain_class: atmo_calc.TerrainClassifier=None):
        self.name = name
        self.booms = {int(boom.height) : boom for boom in booms} # list of booms
        self.latitude = latitude # coord, deg
        self.longitude = longitude # coord, deg
        self.heights = list(self.booms.keys()) # heights at which data is available
        self.terrain_class = terrain_class # terrain classification function

    def save(self, filename: str):
        pass
            
    def remove_outliers(self, n_samples: int=30, sigma: float=5., verbose: bool=False) -> None:
        for boom in self.booms.values():
            boom.remove_outliers(n_samples=n_samples, sigma=sigma, inplace=True, verbose=verbose)

    def resample(self, n_samples: int=10, verbose: bool=False) -> None:
        for boom in self.booms.values():
            boom.resample(n_samples=n_samples, inplace=True, verbose=verbose)

    def classify_terrain(self, height):
        pass

    def compute_vpt(self, verbose: bool=False) -> None:
        for boom in self.booms.values():
            if boom.vpt_computable():
                boom.compute_vpt()
                if verbose:
                    print(f'Computed virtual potential temperatures @{boom.hid}m.')

    @staticmethod
    def load(filename: str) -> MetTower:
        pass

# note to self: use .zst compression
# need a way to account for shadowing in Boom.merge (will look @ hudson's code)
# add functionality to convert Boom winds between polar and cartesian (methods would need to account for this)?
