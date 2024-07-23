### autocorr.py ###
# Elliott Walker #
# 4 Jul 2024 #
# sample analysis of low- vs high-resolution data #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

k = 0.4 # von Karman constant. Between 0.35-0.42, some use 0.35 while Stull uses 0.4
z = 2. # sensor at 2 meters
g = 9.8 # gravitational acceleration
# note - taking theta_v ~ T locally (set reference = values @ 2 m)

# coefficients for dimensionless wind shear computation
ALPHA = 4.7
BETA = 15. # alternatively ~16

def obukhov_length(u_star, T_bar, Q_0):
    # u_start is friciton velocity, T_bar is mean temperature,
    # Q_0 is mean turbulent vertical heat flux
    L = -u_star**3 * T_bar / (k * g * Q_0)
    print(f'\tObukhov length: L = {L:.4f} m')
    return L

def phi(z_over_L):
    # Dimensionless wind shear function
    if z_over_L >= 0: # stable
        # case z/L == 0 is neutral, returns 1 in either formula
        return 1 + ALPHA * z_over_L
    # otherwise, unstable
    return (1 - BETA * z_over_L)**(-1/4)

def wind_gradient(u_star, T_bar, Q_0):
    # uses Businger-Dyer relationship to estimate the vertical gradient of horizontal wind speed, du/dz
    # assume u is aligned with the mean horizontal wind direction
    L = obukhov_length(u_star, T_bar, Q_0)
    grad = u_star / (k * z) * phi(z / L)
    return grad

def flux_richardson(eddy_momt_flux, mean_T, eddy_heat_flux, u_star):
    return (g / mean_T) * eddy_heat_flux / (eddy_momt_flux * wind_gradient(u_star, mean_T, eddy_heat_flux))

def run_analysis(df, name, neglect_subsidence = True):
    print(f'\n*Final analysis: {name} data*\n')
    mean_u = np.mean(df['u'])
    df["u'"] = df['u'] - mean_u
    df["v'"] = df['v']
    df.drop(columns=['v'], inplace=True)
    mean_w = np.mean(df['w'])
    print(f'True mean w (subsidence) is {mean_w:.4f} m/s.')
    if neglect_subsidence:
        print("Neglecting subsidence, so assuming w = w'.")
        df["w'"] = df['w']
        df.drop(columns=['w'], inplace=True) # may be unnecessary/harmful, so could later remove. look back at.
    else:
        df["w'"] = df['w'] - mean_w
    print(f'By definition, mean v is 0. Mean u is {mean_u:.4f} m/s.')
    mean_T = np.mean(df['T'])
    print(f'Mean temperature is {mean_T:.4f} K.')
    df["T'"] = df['T'] - mean_T
    df["w'T'"] = df["w'"] * df["T'"]
    mean_eddy_heat_flux = np.mean(df["w'T'"])
    print(f"Mean eddy heat flux: w'T' = {mean_eddy_heat_flux:.4f} K m/s")
    df["w'u'"] = df["w'"] * df["u'"]
    mean_eddy_momt_flux = np.mean(df["w'u'"])
    print(f"Mean eddy momentum flux: w'u' = {mean_eddy_momt_flux:.4f} m^2/s^2")
    df["w'v'"] = df["w'"] * df["v'"]
    v_flux_part = np.mean(df["w'v'"])
    u_star = (v_flux_part**2 + mean_eddy_momt_flux**2)**(1/4)
    print(f"Friction velocity: u* = {u_star:.4f} m/s")
    ri_f = flux_richardson(mean_eddy_momt_flux, mean_T, mean_eddy_heat_flux, u_star)
    print(f'Flux richardson number: {ri_f:.4f}')

def main(indir, neglect_subsidence = True):
    # read in data
    df_fast = pd.read_csv(f'{indir}/field01_20hz.csv', names=['u','v','w','T'])
    df_slow = pd.read_csv(f'{indir}/field01_minute.csv', names=['u','v','w','T'])

    # convert temperatures to K
    df_fast['T'] += 273.15
    df_slow['T'] += 273.15

    # period of data collection for each set
    PERIOD_SLOW = 60.
    PERIOD_FAST = 0.05

    # time in seconds, relative to start point of data collection
    df_fast['time'] = np.arange(PERIOD_FAST,PERIOD_FAST*(1+len(df_fast)),PERIOD_FAST)
    df_slow['time'] = np.arange(PERIOD_SLOW,PERIOD_SLOW*(1+len(df_slow)),PERIOD_SLOW)

    # now, this time we'll do a change of coordinates:
    # change so that the x (-->u) direction is aligned with the mean wind (y -->v is crosswind)
    print('Aligning coordinate system with x along mean wind...')
    fast_mean_u = np.mean(df_fast['u'])
    slow_mean_u = np.mean(df_slow['u'])
    fast_mean_v = np.mean(df_fast['v'])
    slow_mean_v = np.mean(df_slow['v'])
    fast_phi = np.arctan2(fast_mean_v, fast_mean_u)
    fast_mean_dir = (np.rad2deg(fast_phi) + 360) % 360
    print(f'\tFast data aligned to {fast_mean_dir:.4f} degrees.')
    slow_phi = np.arctan2(slow_mean_v, slow_mean_u)
    slow_mean_dir = (np.rad2deg(slow_phi) + 360) % 360
    print(f'\tSlow data aligned to {slow_mean_dir:.4f} degrees.')
    df_fast_corrected = df_fast.copy()
    df_slow_corrected = df_slow.copy()
    df_fast_corrected['u'] = df_fast['u'] * np.cos(fast_phi) + df_fast['v'] * np.sin(fast_phi)
    df_fast_corrected['v'] = -df_fast['u'] * np.sin(fast_phi) + df_fast['v'] * np.cos(fast_phi)
    df_fast_corrected.set_index('time', inplace=True)
    df_slow_corrected.set_index('time', inplace=True)

    # run analysis script
    run_analysis(df_fast_corrected, 'fast', neglect_subsidence)
    run_analysis(df_slow_corrected, 'slow', neglect_subsidence)

if __name__ == '__main__':
    print('WITH SUBSIDENCE NEGLECTED:')
    main('data', True)
    print('\n\n\nWITHOUT SUBSIDENCE NEGLECTED:')
    main('data', False)
