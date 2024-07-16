import pandas as pd
import numpy as np
import helper_functions as hf

# The CSV data in this folder does not include sonic anemometer data: wind speed/direction is all from the cup anemometer+vane.

# Sampling rate is 1Hz, logging period is 1 minute (mean average taken each period)

# 1: 6m, 2: 10m, 3: 20m, 4: 32m, 5: 80m, 6: 106m, 7: 106m

# Wind speeds: m/s      Wind directions: deg CW of N    Temperatures: deg C -> K (+273.15)     Pressures: mmHg -> kPa (*0.13332239)

# Manual changes made to data: removed -2 at end of a few file names; changed Boom1 dates from xx -> 20xx so pd can process as datetimes

SPEED_THRESHOLD = 0.1 # minimum speed threshold of wind speed sensors
clear_threshold = False # replace wind direction where speed is below this threshold to NaN, and speed to 0

parent = '../../DATA/CedarRapids'

boom1 = pd.read_csv(f'{parent}/Boom1OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_6m',
                                                       'MeanDirection' : 'wd_6m',
                                                       'MeanTemperature (C )' : 't_6m',
                                                       'MeanPressure (mmHg)' : 'p_6m'})

boom2 = pd.read_csv(f'{parent}/Boom2OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_10m',
                                                       'MeanDirection' : 'wd_10m',
                                                       'MeanTemperature (C )' : 't_10m',
                                                       'MeanRH (%)' : 'rh_10m'})

boom3 = pd.read_csv(f'{parent}/Boom3OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_20m',
                                                       'MeanDirection' : 'wd_20m'})

boom4 = pd.read_csv(f'{parent}/Boom4OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity' : 'ws_32m',
                                                       'MeanDirection' : 'wd_32m',
                                                       'MeanTemperature' : 't_32m',
                                                       'MeanRH' : 'rh_32m'})

boom5 = pd.read_csv(f'{parent}/Boom5OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity' : 'ws_80m',
                                                       'MeanDirection' : 'wd_80m',
                                                       'MeanTemperature' : 't_80m',
                                                       'MeanRH' : 'rh_80m'})

boom6 = pd.read_csv(f'{parent}/Boom6OneMin.csv').rename(columns={'TIMESTAMP' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_106m1',
                                                       'Mean Direction' : 'wd_106m1',
                                                       'MeanTemperature (C )' : 't_106m',
                                                       'MeanRH (%)' : 'rh_106m'})

boom7 = pd.read_csv(f'{parent}/Boom7OneMin.csv').rename(columns={'TimeStamp' : 'time',
                                                       'MeanVelocity (m/s)' : 'ws_106m2',
                                                       'MeanDirection' : 'wd_106m2',
                                                       'MeanPressure (mmHg)' : 'p_106m'})

# Datetime conversion
# Note - times are all UTC
boomlist = [boom1,boom2,boom3,boom4,boom5,boom6,boom7]
for boom in boomlist:
    boom['time'] = pd.to_datetime(boom['time'])
    boom.set_index('time', inplace=True)

# Merge all frames together using an inner merge, except for boom 5 (left merge instead)
df = boom1.merge(boom2, on='time', how='inner').merge(boom6, on='time', how='inner').merge(boom7, on='time', how='inner').merge(boom3, on='time',how='left').merge(boom4, on='time', how='left').merge(boom5, on='time', how='left')

# take the vector mean of the wind between the two booms at 106m
df['ws_106m'], df['wd_106m'] = hf.combine_top(df)
df.drop(columns=['ws_106m1','ws_106m2','wd_106m1','wd_106m2'],inplace=True)
df.sort_values(by='time',ascending=True,inplace=True)

for column in df.columns:
    if 'p_' in column:
        # convert pressures from mmHg to kPa
        df[column] = df[column] * 0.13332239
    elif 't_' in column:
        # convert temperatures from deg C to K
        df[column] = df[column] + 273.15
    elif 'rh_' in column:
        # convert relative humidity from % to decimal
        df[column] = df[column] / 100.
    elif 'ws_' in column:
        # when wind speeds are 0, set the direction to nan
        # choosing not to eliminate those below the measurement threshold for now
        dircol = 'wd_' + column.split('_')[1][:-1] + 'm'
        df.loc[df[column] == 0, dircol] = np.nan
        if clear_threshold:
            df.loc[df[column] < SPEED_THRESHOLD, dircol] = np.nan
            df.loc[df[column] < SPEED_THRESHOLD, column] = 0
    elif 'wd_' in column:
        df[column] = (df[column]-90) % 360
    if column != 'time':
        #sensors only have so much precision, so we can save half of the memory by casting down to a lower-precision float
        #don't cast any lower: casting to float16 does lead to significant roundoff error at some later stages
        df[column] = df[column].astype('float32')

print(f'Exporting combined data. Length: {len(df)} rows.')

df.to_csv('combined.csv')
