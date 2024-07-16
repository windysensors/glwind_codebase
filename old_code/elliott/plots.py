# primary generation of plots

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import helper_functions as hf

df10 = pd.read_csv('ten_minutes_labeled.csv') # 10-minute averaged data, with calculations and labeling performed by reduce.py
df10['time'] = pd.to_datetime(df10['time'])

# Useful list of all of the heights, in m, that data exists at
heights = [6,10,20,32,80,106]

def scatter_ri():
    # Plot of Ri over time
    plt.scatter(df10['time'], df10['ri'],s=0.1)
    plt.show()
    return

def bar_stability(combine=False):
    # Bar chart of stability classifications
    stability_r_freqs = df10['stability'].value_counts(normalize=True)
    if combine:
        plt.bar(['unstable','neutral','stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable']+stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange'])
    else:
        plt.bar(['unstable','neutral','stable','strongly stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable'],stability_r_freqs['strongly stable']], color=['mediumblue','deepskyblue','orange','crimson'])
    plt.ylabel('relative frequency')
    plt.title('wind data sorted by thermal stability classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    plt.show()
    return

def bar_new(save=False):
    # Bar chart of stability classifications
    stability_r_freqs = df10['new_stability'].value_counts(normalize=True)
    plt.bar(['unstable','neutral','stable'],[stability_r_freqs['unstable'],stability_r_freqs['neutral'],stability_r_freqs['stable']], color=['mediumblue','deepskyblue','orange'])#,'crimson'])
    plt.ylabel('relative frequency')
    plt.title('frequency of wind data sorted by new stability classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    if save:
        plt.savefig('plots/week3pres/newbar.png')
    else:
        plt.show()
    return

def bar_directions():
    # Bar chart of direction classifications
    dir_r_freqs = df10['terrain'].value_counts(normalize=True)
    plt.bar(dir_r_freqs.index, dir_r_freqs.values, color=['red','green','blue'])
    plt.ylabel('relative frequency')
    plt.title('frequency of wind data sorted by terrain classification')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='-', alpha=0.8)
    plt.show()
    return

def scatter_rat():
    # Plot of vpt_lapse_env over time
    plt.scatter(df10['time'], df10['vpt_lapse_env'],s=0.2)
    plt.xlabel('time')
    plt.ylabel(r'$\Delta \theta_{v}/\Delta z$, K/m')
    plt.show()
    return

def plot_temp(height=6):
    # plot of temperature at 6m over time
    plt.scatter(df10['time'], df10[f't_{height}m'],s=1)
    plt.show()
    return

def alpha_vs_lapse(limit=True):
    df = df10.dropna(subset=['vpt_lapse_env','alpha'],how='any')
    plt.scatter(df['vpt_lapse_env'],df['alpha'],s=0.5)
    if limit:
        plt.xlim([-0.025,0.1])
        plt.ylim([-0.3,1.25])
    plt.xlabel(r'$\Delta \theta_{v}/\Delta z$, K/m')
    plt.ylabel(r'$\alpha$')
    corr = np.corrcoef(df['vpt_lapse_env'], df['alpha'])[0,1]
    plt.title(r'$r={{{r:.4f}}}$'.format(r=corr, r2=corr**2))
    plt.show()
    return

def alpha_vs_ri():
    plt.scatter(df10['ri'],df10['alpha'],s=0.5)
    plt.xlim([-35,25])
    plt.ylim([-0.3,1.25])
    plt.xlabel('Ri')
    plt.ylabel(r'$\alpha$')
    plt.show()
    return

def plot_alpha():
    plt.scatter(df10['time'],df10['alpha'],s=0.4)
    plt.scatter(df10['time'],df10['t_10m']/50-5, s=0.3)
    plt.xlabel('time')
    plt.ylabel(r'$\alpha$')
    plt.show()
    return

if __name__ == '__main__':
    bar_stability()
