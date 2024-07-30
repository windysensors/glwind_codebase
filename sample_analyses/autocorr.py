### autocorr.py ###
# Elliott Walker #
# 17 Jul 2024 #
# sample sonic data analysis #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

def ANALYZE(df, *, integral = False, threshold=0.25):
    scales = {}
    for var in df.columns:
        print(f'Autocorrelating {var}')
        Ruu = []
        kept = len(df) // 2
        lost = len(df) - kept
        for i in tqdm(range(kept)):
            autocorr = df[var].autocorr(lag=i)
            Ruu.append(autocorr)
        df[f'Ruu_{var}'] = Ruu + [np.nan] * lost
        if (not integral) or (var == 'T'):
            continue
        dt = df.index[1] - df.index[0]
        mean_speed = df[var].mean()
        cutoff_index = 0
        for i, val in enumerate(Ruu):
            if val < threshold:
                cutoff_index = i
                break
        if cutoff_index == 0:
            print('Warning: failed to find cutoff for integration')
        i_time = np.sum(Ruu[:cutoff_index]) * dt
        i_length = i_time * mean_speed
        print(f'Mean wind speed [{var}]: {mean_speed:.3f} m/s')
        print(f'Integral time scale [{var}]: {i_time:.3f} s')
        print(f'Integral length scale [{var}]: {i_length:.3f} m')
        scales[var] = (i_time, i_length)
    return scales

VARLOOKUP = {
    'u' : ('wind speed', 'm/s'),
    'v' : ('wind speed', 'm/s'),
    'w' : ('wind speed', 'm/s'),
    'T' : ('temperature', 'K')
}

def PLOT(df, *, name='data', show=False, integral_scales=None):
    for var in df.columns:
        if 'Ruu' in var:
            continue
        dimn, unit = VARLOOKUP[var]
        mean = df[var].mean()
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.suptitle(f'{name.capitalize()} data - {dimn.capitalize()} [{var}]', fontweight='bold')
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel(f'{dimn} ({unit})')
        axs[0].plot(df.index, df[var], c='tab:blue', label=str(var), linewidth=1)
        axs[0].plot(df.index, [mean]*len(df), c='tab:red', label='mean')
        axs[0].legend()
        if integral_scales and var in integral_scales.keys():
            axs[1].set_title(f'Integral scales: time={integral_scales[var][0]:.3f}, length={integral_scales[var][1]:.3f}')
        axs[1].set_xlabel('lag (s)')
        axs[1].set_ylabel('autocorrelation')
        axs[1].plot(df.index, df[f'Ruu_{var}'], c='tab:blue', linewidth=1)
        axs[1].plot(df.index, [0]*len(df), c='black', linestyle='dashed')
        fig.tight_layout(pad=1)
        if show:
            plt.show()
        else:
            plt.savefig(f'plots/{name}_{var}.png',bbox_inches='tight')
        plt.close()
    return

def load_data(filepath, kelvinConvert = True):
    df = pd.read_csv(filepath).set_index('t')
    if kelvinConvert and 'T' in df.columns:
        df['T'] += 273.15
    return df

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    df_sample = load_data('data/short_sample.csv')
    df_slow = load_data('data/field01_minute.csv')
    df_fast = load_data('data/field01_20hz.csv')
    print('Sample data:')
    ANALYZE(df_sample)
    PLOT(df_sample, name='sample')
    print('Slow data:')
    ANALYZE(df_slow)
    PLOT(df_slow, name='slow')
    print('Fast data:')
    integral_scales = ANALYZE(df_fast, integral = True)
    PLOT(df_fast, name='fast', integral_scales=integral_scales)
