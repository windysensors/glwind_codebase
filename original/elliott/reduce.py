# reduce, resample, compute, classify
# outlier removal, 10 minute resampling, value calculations & classifications

import pandas as pd
import numpy as np
import helper_functions as hf

# Read in dataframe created in combine.py
df = pd.read_csv('combined.csv') # File from combine.py; this has the data overlapping from all booms, except 5 is only where available
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Useful list of all of the heights, in m, that data exists at
heights = [6,10,20,32,80,106]

# Eliminate data which is is more than 5 stds from mean in 30 minute window
# Don't worry about direction for now...
eliminations = 0
dont_consider = [f'wd_{h}m' for h in heights] + ['time']
to_consider = [col for col in df.columns.values.tolist() if col not in dont_consider]
for column in to_consider:
    rolling_mean = df[column].rolling(window='30T').mean()
    rolling_std = df[column].rolling(window='30T').std()
    threshold = 5 * rolling_std
    outliers = np.abs(df[column] - rolling_mean) > threshold
    eliminations += df[outliers].shape[0]
    df = df[~outliers]

print(f'{eliminations} outliers eliminated ({100*eliminations/(df.shape[0]+eliminations):.4f}%)')

# Now, we want to combine the data into 10-minute averages.
for h in heights:
    dirRad = np.deg2rad(df[f'wd_{h}m'])
    df[f'x_{h}m'] = df[f'ws_{h}m'] * np.sin(dirRad)
    df[f'y_{h}m'] = df[f'ws_{h}m'] * np.cos(dirRad)
df_10_min_avg = df.resample('10min').mean()
before_dropna = len(df_10_min_avg)
df_10_min_avg.dropna(axis=0,how='all',inplace=True) # If any row is completely blank, drop it
dropped = before_dropna - (len(df_10_min_avg))
if dropped > 0:
    print(f'{dropped} blank row(s) removed')
for h in heights:
    df_10_min_avg[f'ws_{h}m'] = np.sqrt(df_10_min_avg[f'x_{h}m']**2+df_10_min_avg[f'y_{h}m']**2)
    df_10_min_avg[f'wd_{h}m'] = (np.rad2deg(np.arctan2(df_10_min_avg[f'x_{h}m'], df_10_min_avg[f'y_{h}m'])) + 360) % 360
    df_10_min_avg.drop(columns=[f'x_{h}m',f'y_{h}m'],inplace=True)

# Compute virtual potential temperatures at 10 and 106 meters
df_10_min_avg['vpt_106m'] = df_10_min_avg.apply(lambda row: hf.virtual_potential_temperature(row['rh_106m'],row['p_106m'],row['t_106m']), axis=1).astype('float32')
df_10_min_avg['vpt_10m'] = df_10_min_avg.apply(lambda row: hf.virtual_potential_temperature(row['rh_10m'],row['p_6m'],row['t_10m']), axis=1).astype('float32')

# Compute bulk Richardson number
df_10_min_avg['ri'] = df_10_min_avg.apply(lambda row: hf.bulk_richardson_number(row['vpt_10m'],row['vpt_106m'],10.,106.,row['ws_10m'],row['ws_106m'],row['wd_10m'],row['wd_106m']), axis=1).astype('float32')

# Measure of average change in virtual temperature per meter over full height (delta virtual potential temperature / delta z)
df_10_min_avg['vpt_lapse_env'] = (df_10_min_avg['vpt_106m'] - df_10_min_avg['vpt_10m'])/96.

# New stability classification scheme
# df_10_min_avg['new_stability'] = df_10_min_avg.apply(lambda row: hf.new_class(row['vpt_lapse_env']), axis=1).astype('category')

# Drop any rows without Richardson number calculated
df_10_min_avg.dropna(subset=['ri'],inplace=True)

# Stability classification based on Ri
df_10_min_avg['stability'] = df_10_min_avg.apply(lambda row: hf.stability_class(row['ri']), axis=1).astype('category')

# Terrain classification based on wind direction at 10m
df_10_min_avg['terrain'] = df_10_min_avg.apply(lambda row: hf.terrain_class(row['wd_10m']), axis=1).astype('category')

# Wind shear exponent
df_10_min_avg['alpha'] = df_10_min_avg.apply(lambda row: hf.power_fit(heights,[row[f'ws_{h}m'] for h in heights]), axis=1)

print(f'Exporting reduced data. Length: {len(df_10_min_avg)} rows.')

# Save to CSV
df_10_min_avg.to_csv('ten_minutes_labeled.csv')

