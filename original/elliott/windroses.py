import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from windrose import WindroseAxes
import os
import math

# Default bins
DEFAULT_BINS = [0.89,2.24,3.13,4.47,6.71,8.94]
# Heights at which data is found, ignoring 80m because we want a good year-round picture
HEIGHTS = [6,10,20,32,106]

# Month dictionary, as N : name
# For abbreviation just use months[N][:3]
months = {1 : 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

class InsufficientError(Exception):
    pass

def month_rose(df, month, which_heights=HEIGHTS, bins=DEFAULT_BINS, saveto=None, transparent=False):
    # Uses data from a specific month to generate a windrose
    mname = months[month]
    dfm = df10[df10['time'].dt.month == month]
    speeds = []
    dirs = []
    for h in which_heights:
        speeds += list(dfm[f'ws_{h}m'])
        dirs += list(dfm[f'wd_{h}m'])
    nans = 0
    for s in speeds:
        if math.isnan(s):
            nans += 1
    if nans >= len(speeds)-10:
        raise InsufficientError
    ax = WindroseAxes.from_ax()
    ax.bar(dirs, speeds, normed=True, opening=1.0, bins=bins, edgecolor='white', cmap=cm.rainbow, nsector=36)
    ax.set_legend()
    heightstr = ''
    if len(which_heights) == 1:
        heightstr = f' ({which_heights[0]}m)'
    elif which_heights == HEIGHTS:
        heightstr = f' (all heights)'
    plt.title(f'Cedar Rapids, Iowa: {mname}{heightstr}')
    plt.text(1.04, -0.04, 'Based on data from Sept 2017 - Sept 2018', ha='right', va='bottom', transform=ax.transAxes)
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto, bbox_inches='tight', transparent=transparent)
    else:
        plt.show()
    plt.close()
    return

def full_rose(df, which_heights=HEIGHTS, bins=DEFAULT_BINS, saveto=None, transparent=False):
    # Uses data over all time to generate a windrose
    speeds = []
    dirs = []
    for h in which_heights:
        speeds += list(df[f'ws_{h}m'])
        dirs += list(df[f'wd_{h}m'])
    ax = WindroseAxes.from_ax()
    ax.bar(dirs, speeds, normed=True, opening=1.0, bins=bins, edgecolor='white', cmap=cm.rainbow, nsector=36)
    ax.set_legend()
    heightstr = ''
    if len(which_heights) == 1:
        heightstr = f' ({which_heights[0]}m)'
    elif which_heights == HEIGHTS:
        heightstr = ' (all heights)'
    plt.title(f'Cedar Rapids, Iowa{heightstr}')
    plt.text(1.04, -0.04, 'Based on data from Sept 2017 - Sept 2018', ha='right', va='bottom', transform=ax.transAxes)
    if saveto is not None:
        os.makedirs(os.path.dirname(saveto), exist_ok=True)
        plt.savefig(saveto, bbox_inches='tight', transparent=transparent)
    else:
        plt.show()
    plt.close()
    return

def generate_roses(df, directory='plots/windroses', transparent=False):
    print('Generating all windroses.')
    print('Year round...')
    fulldir = directory+'/all/'
    full_rose(df, saveto=fulldir+'rose_full.png', transparent=transparent)
    for h in HEIGHTS:
        full_rose(df, [h], saveto=fulldir+f'rose_{h}m.png', transparent=transparent)
    for n, mname in months.items():
        print(f'{mname}...')
        subdir = directory+f'/{mname}/'
        month_rose(df, n, saveto=subdir+'rose_full.png', transparent=transparent)
        for h in HEIGHTS:
            try:
                month_rose(df, n, [h], saveto=subdir+f'rose_{h}m.png', transparent=transparent)
            except InsufficientError:
                print(f'Insufficient data to generate {mname} rose at {h}m.')
            except:
                print(f'Could not generate {mname} rose at {h}m.')
    print('Complete!')
    return

if __name__ == '__main__':
    df10 = pd.read_csv('ten_minutes_labeled.csv') # 10-minute averaged data, with calculations and labeling performed by reduce.py
    df10['time'] = pd.to_datetime(df10['time'])
    generate_roses(df10)
