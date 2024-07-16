### booms.py ###
# author: Elliott Walker
# last update: 11 July 2024
# description: Boom and MetTower object classes

from __future__ import annotations
from units import Quantity, defaults
from units import convert_value as cnv
import stat_calc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import atmo_calc
import traceback
import roses
import os
import sys
from tqdm import tqdm
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

# add images as textures in 3d space
def _add_image(ax, img_path, height, z_offset):
    img = mpimg.imread(img_path)
    img_height, img_width, _ = img.shape

    x = np.linspace(-img_width/200, img_width/200, img_width)
    y = np.linspace(-img_height/200, img_height/200, img_height)
    x, y = np.meshgrid(x, y)
    z = np.full_like(x, z_offset)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=img, shade=False)
    
    return

# Boom object class
# Structure for holding time-series data from a single met tower boom (pd.DataFrame underlying)
# Methods for merging with other Booms, as well as data handling (outlier removal, resampling)
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

    def copy(self, keep_notes: bool=True) -> Boom:
        return Boom(self._data.copy(), self.height, automatic = True, notes = self.notes * keep_notes)

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
            print(f'Outlier elimination performed on boom @{self.hid}m.')
            if eliminations > 0:
                print(f'\t{eliminations} outliers eliminated ({100*eliminations/(df.shape[0]+eliminations):.4f}%)')
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
            print(note)
            if eliminations > 0:
                print(f'\t{eliminations} blank row(s) removed')
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

# MetTower object class
# Structure for holding a collection of booms
class MetTower:
    def __init__(self, name: str, booms: list[Boom]|dict[str,Boom], latitude: float=None, longitude: float=None, terrain_class: atmo_calc.TerrainClassifier=None,
                 copy_booms: bool=True, warnings: bool=True):
        def boomTypeErr(): raise TypeError("'booms' must be a list of Boom objects or a dict of name:Boom pairs")
        self.name = name
        if type(booms) is list: # if a list is passed, assign identifiers based on Boom height id (hid)
            for boom in booms: # make sure booms are all of Boom type
                if type(boom) is not Boom:
                    boomTypeErr()
            if copy_booms:
                self.booms = {str(boom.hid) : boom.copy() for boom in booms}
            else:
                self.booms = {str(boom.hid) : boom for boom in booms}
            if len(self.booms) < len(booms) and warnings:
                print("Warning: data loss due to overlapping boom height identifiers.\n\tIf loss was not intentional, try instead passing 'booms' as a dictionary of name:Boom pairs.")
        elif type(booms) is dict:
            for boom in booms.values(): # make sure booms are all of Boom type
                if type(boom) is not Boom:
                    boomTypeErr()
            if copy_booms:
                self.booms = {str(name) : boom.copy() for name, boom in booms.items()}
            else:
                self.booms = {str(name) : boom for name, boom in booms.items()} # still uses dict comprehension because we want to ensure that the keys are strings
        else: boomTypeErr()
        self.latitude = latitude # coord, deg
        self.longitude = longitude # coord, deg
        self.available_heights = [boom.height for boom in self.booms.values()].sort()
        self.terrain_class = terrain_class # terrain classification function
        self.has_canonical_time = False

    def save(self, filename: str):
        pass
            
    def remove_outliers(self, n_samples: int=30, sigma: float=5., verbose: bool=False) -> None:
        # bulk processing: remove outliers in all booms
        for boom in self.booms.values():
            boom.remove_outliers(n_samples=n_samples, sigma=sigma, inplace=True, verbose=verbose)

    def resample(self, n_samples: int=10, verbose: bool=False) -> None:
        # bulk processing: resample all booms
        for boom in self.booms.values():
            boom.resample(n_samples=n_samples, inplace=True, verbose=verbose)

    def associate_canonical_time(self, which_booms: list[Boom], verbose: bool=False, warnings: bool=True) -> None:
        # associate a canonical datetime series for this MetTower, which is the series contanining datetimes common to all booms specified in <which_booms> (inner merge of their time series indices)
        if self.has_canonical_time:
            print(f"MetTower '{self.name}' already has a canonical time associated with it.\n\tTo unassociate, make a call to MetTower.unassociate().")
            return
        timeSeries = None
        for boom in which_booms:
            if boom not in self.booms.keys():
                if warnings:
                    print(f"Warning: in canonical time association for MetTower '{self.name}', could not find Boom '{boom}' - skipping.")
                continue
            boomTime = self.booms[boom]._data.reset_index()['time'] # or just _data.index ?
            if timeSeries is not None:
                timeSeries = timeSeries[timeSeries.isin(boomTime)]
            else:
                timeSeries = boomTime
        if timeSeries is None:
            return
        self._data = pd.DataFrame(timeSeries, columns=['time'])
        if verbose:
            print(f"Associated canonical datetimes (N = {len(self._data)}) to MetTower '{self.name}'.")
        #print(self._data)
        self.has_canonical_time = True

    def unassociate_canonical_time(self) -> None:
        # unassociate the canonical datetime series associated by MetTower.associate_canonical_time
        if not self.has_canonical_time:
            print(f"MetTower '{self.name}' does not have an associated canonical time.")
            return
        del self._data
        self.has_canonical_time = False

    def classify_terrain(self, boom: str):
        # determine terrain classification based on the TerrainClassifier object associated with this MetTower
        if boom not in self.booms.keys():
            raise NameNotFound(f"Boom '{boom}' not found in MetTower {self.name}.")
        boomObj = self.booms[boom]
        if 'wd' not in boomObj.available_data:
            raise AvailabilityError(f"Boom '{boom}' in MetTower {self.name} does not contain wind direction data.")
        directions = boomObj['wd']
        pass

    def compute_ri(self, boom1: str, boom2: str, verbose: bool=False):
        # computes bulk Richardson number `ri` based on data at two booms
        if not self.has_canonical_time:
            raise AvailabilityError(f"Could not compute bulk Ri for MetTower '{self.name}': no associated canonical time.")
        df1 = self.booms[boom1]._data
        df2 = self.booms[boom2]._data
        for bname, df in [(boom1, df1), (boom2, df2)]:
            if 'vpt' not in df.columns:
                if self.booms[bname].vpt_computable():
                    df.compute_vpt()
                    print(f"Computed missing virtual potential temperatures for boom '{bname}'")
                else:
                    raise AvailabilityError(f"Could not compute bulk Ri for MetTower '{self.name}': virtual potential temperature missing & not computable for boom '{bname}'")
        dfJoined = pd.merge(df1[['vpt','ws','wd']], df2[['vpt','ws','wd']], how='inner', on='time', suffixes=('_1', '_2')).merge(self._data['time'], how='inner', on='time')
        self._data['ri'] = dfJoined.apply(lambda row : atmo_calc.bulk_richardson_number(row['vpt_1'], row['vpt_2'], self.booms[boom1].height, self.booms[boom2].height, row['ws_1'], row['ws_2'], row['wd_1'], row['wd_2']), axis=1)
        eliminations = self._data['ri'].isna().sum()
        self._data.dropna(subset=['ri'], inplace=True)
        if verbose:
            print(f"Computed bulk Richardson numbers for MetTower '{self.name}'.")
            print(f"\tDropped {eliminations} times with NaN Ri output.")

    def classify_stability(self, verbose: bool=False):
        # determines stability classification using bulk Richardson number, using the function atmo_calc.bulk_stability_class
        if not self.has_canonical_time:
            raise AvailabilityError(f"Could not classify stability for MetTower '{self.name}': no associated canonical time.")
        if not 'ri' in self._data.columns:
            raise AvailabilityError(f"Could not classify stability for MetTower '{self.name}': bulk Ri not computed.")
        self._data['stability'] = self._data.apply(lambda row : atmo_calc.bulk_stability_class(row['ri']), axis=1)
        if verbose:
            print(f"Determined stability classifications for MetTower '{self.name}'.")

    def save_stabilities(self, filename: str, combine: bool=False, verbose: bool=False):
        # save a bar plot of the relative frequencies of the stability classifications
        if not self.has_canonical_time:
            raise AvailabilityError(f"Could not plot stabilities for MetTower '{self.name}': no associated canonical time.")
        if not 'stability' in self._data.columns:
            raise AvailabilityError(f"Could not plot stabilities for MetTower '{self.name}': stability classifications not determined.")
        stability_r_freqs = self._data['stability'].value_counts(normalize=True)
        if combine:
            plt.bar(['unstable','neutral','stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable']+stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange'])
        else:
            plt.bar(['unstable','neutral','stable','strongly stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable'],stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange','crimson'])
        plt.ylabel('relative frequency')
        plt.title('wind data sorted by thermal stability classification')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='-', alpha=0.8)
        plt.savefig(filename)
        plt.close()
        
    def compute_powerlaws(self, canon: bool=False):
        pass

    def compute_averages(self):
        pass

    def compute_average_powerlaws(self):
        pass

    def compute_vpt(self, verbose: bool=False) -> None:
        for boom in self.booms.values():
            if boom.vpt_computable():
                boom.compute_vpt()
                if verbose:
                    print(f'Computed virtual potential temperatures @{boom.hid}m.')

    def set_coordinates(self, latitude: float=None, longitude: float=None) -> None:
        if latitude:
            self.latitude = latitude
        if longitude:
            self.longitude = longitude

    def save_windrose(self, boom: str, filename: str, mode: str='speed', palette=None, canon: bool=False, bin_size: int=None, N_bins: int=None, verbose: bool=False, warnings: bool=True):
        if boom == 'all':
            return # FIX
        if mode.lower() == 'speed':
            df = self.booms[boom]._data[['ws','wd']]
            if canon:
                if not self.has_canonical_time:
                    raise AvailabilityError(f"Could not plot speed windrose for MetTower '{self.name}' with setting canon=True: no associated canonical time.")
                before_elim = len(df)
                df = pd.concat([df, self._data.set_index('time')], axis=1, join='inner')
                if verbose:
                    print(f'{before_elim - len(df)} points ignored due to setting canon=True.')
        elif mode.lower() == 'ri':
            if canon and warnings:
                print("Warning: setting canon=True is redundant, as mode='ri' already takes canonical time.")
            if not self.has_canonical_time:
                raise AvailabilityError(f"Could not plot Ri windrose for MetTower '{self.name}': no associated canonical time.")
            if not 'ri' in self._data.columns:
                raise AvailabilityError(f"Could not plot Ri windrose for MetTower '{self.name}': Ri not available.")
            df = pd.concat([self.booms[boom]._data['wd'], self._data.set_index('time')['ri']], axis=1, join='inner')
        else:
            raise ModeNotFound(f"Mode '{mode}' not found for plotting windrose.")
        if bin_size is None and N_bins is None:
            bin_size = 15
        fig = roses.windrose(df, mode=mode, palette=palette, bin_size=bin_size, N_bins=N_bins)
        fig.savefig(filename)
        plt.close(fig)
        if verbose:
            print(f"Saved windrose (boom: {boom}, mode: {mode}) to {filename}.")
    
    def show_windstack(self, which_booms: list[str]=None, canon: bool=False, verbose: bool=False, warnings: bool=True):
        # DOES NOT WORK
        if which_booms is None:
            which_booms = list(self.booms.keys())
        locs = []
        heights = []
        for n, bname in enumerate(which_booms):
            if bname in self.booms.keys():
                loc = f'TEMP_{n}.png'
                self.save_windrose(bname, f'TEMP_{n}.png')
                locs.append(loc)
                heights.append(self.booms[bname].height)
            elif warnings:
                print(f"Warning: in windrose stack-plot for MetTower '{self.name}', could not find Boom '{bname}' - skipping.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        zipd = zip(locs, heights)
        if verbose:
            zipd = tqdm(zipd)
        for loc, height in zipd:
            _add_image(ax, loc, height, height)
            os.remove(loc)

        ax.set_zlabel('z')
        ax.set_zlim(0, max(heights)*1.05)

        plt.show()

    @staticmethod
    def load(filename: str) -> MetTower:
        pass

# note to self: use .zst compression
# need a way to account for shadowing in Boom.merge (will look @ hudson's code)
# add functionality to convert Boom winds between polar and cartesian (methods would need to account for this)?
# should fix atmo_calc.local_gravity and possibly incorporate into bulk Ri calculation
