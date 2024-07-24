# generate profile plots

import helper_functions as hf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# list of heights with data, minus the 80 meter data because it's incomplete
heights = [6,10,20,32,106]
# classes
stability_classes = ['unstable','neutral','stable','strongly stable']
terrain_classes = ['complex','open']

df10 = pd.read_csv('ten_minutes_labeled.csv')

zvals = np.linspace(0.,130.,400)

def plot_full_profile():
    for tc in terrain_classes:
        df = df10[df10['terrain'] == tc]
        plt.xlabel('Mean Velocity (m/s)')
        plt.ylabel('Height (m)')
        plt.title(f'{tc.capitalize()} terrain wind profile')
        for sc in stability_classes:
            df_sc = df[df['stability'] == sc]
            means = df_sc[[f'ws_{h}m' for h in heights]].mean(axis=0)
            mult, wsc = hf.power_fit(heights, means.values, both=True)
            plt.scatter(means.values, heights, label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc,a=mult,b=wsc))
            plt.plot(mult * zvals**wsc, zvals)
            print(f'{tc}, {sc}: mult = {mult:.4f}, alpha = {wsc:.4f}')
        plt.legend()
        plt.show()

def plot_need_boom5():
    old = len(df10)
    df_red = df10.dropna(subset=['ws_80m'])
    new = len(df_red)
    print(f'{old-new}/{old} dropped')
    for tc in terrain_classes:
        df = df_red[df_red['terrain'] == tc]
        plt.xlabel('Mean Velocity (m/s)')
        plt.ylabel('Height (m)')
        plt.title(f'{tc.capitalize()} terrain wind profile')
        for sc in stability_classes:
            df_sc = df[df['stability'] == sc]
            means = df_sc[[f'ws_{h}m' for h in heights+[80]]].mean(axis=0)
            mult, wsc = hf.power_fit(heights+[80], means.values, both=True)
            plt.scatter(means.values, heights+[80], label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc,a=mult,b=wsc))
            plt.plot(mult * zvals**wsc, zvals)
            print(f'{tc}, {sc}: mult = {mult:.4f}, alpha = {wsc:.4f}')
        plt.legend()
        plt.show()

def plot_noterrain():
    fig, ax = plt.subplots(figsize = (5.5,4))
    fig.suptitle('Annual Wind Profiles by Stability Class (Power Law Fits)')
    ax.set_xlabel('Mean Velocity (m/s)')
    ax.set_ylabel('Height (m)')
    for sc in stability_classes:
        df_sc = df10[df10['stability'] == sc]
        means = df_sc[[f'ws_{h}m' for h in heights]].mean(axis=0)
        mult, wsc = hf.power_fit(heights, means.values, both=True)
        ax.scatter(means.values, heights, label=r'{sc}: $u(z)={a:.2f}z^{{{b:.3f}}}$'.format(sc=sc.capitalize(),a=mult,b=wsc))
        ax.plot(mult * zvals**wsc, zvals)
        print(f'{sc}: mult = {mult:.4f}, alpha = {wsc:.4f}')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    plot_noterrain()
    #plot_full_profile()
    #plot_need_boom5()
