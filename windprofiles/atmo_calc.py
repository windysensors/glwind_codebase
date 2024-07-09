### atmo_calc.py ###
# author: Elliott Walker
# last update: 9 July 2024
# description: functions for atmosphere-related calculations

import numpy as np
from exceptions import *

LOCAL_GRAVITY_DEFAULT = 9.802 # local gravity at Cedar Rapids (latitude ~ 42 degrees, elevation ~ 247 m), in m/s^2

# local gravity function - calculates effective local gravity as a function of latitude and elevation
# latitude should be in degrees, elevation in meters
# may be incorrect. see https://www.sensorsone.com/local-gravity-calculator/#height for this formula
#   (current issue: this uses the formula they quote, but our results are different than their implementation)
def local_gravity(latitude: float, elevation: float) -> float:
    IGF = 9.780327 * (1 + 0.0053024 * np.sin(np.deg2rad(latitude))**2 - 0.0000058*np.sin(np.deg2rad(2*latitude))**2)
    FAC = -3.086e-6 * elevation
    return IGF + FAC

# saturation vapor pressure - calculates SVP in kPa using Tetens' approximation
# temperature T should be in K
def saturation_vapor_pressure(T: float) -> float:
    return 0.6113 * np.exp(17.2694 * (T - 273.15) / (T - 35.86))

# virtual potential temperature - calculates v.p.t. in K
def virtual_potential_temperature(RH: float, P: float, T: float, approx: bool=False) -> float:
    # given RH in [0,1], P in kPa, T in K, computes v.p.t.
    e_s = saturation_vapor_pressure(T)
    e = RH * e_s # actual vapor pressure
    mwr = 0.622 # molecular weight ratio of water to air
    w = mwr * e / (P - e) # mixing ratio
    P0 = 100. # reference pressure in kPa
    vT = T * (P0/P)**0.286 # virtual temperature
    if approx:
        # first order approximation of the exact formula
        # valid within ~1% for mixing ratios between roughly 0.00-0.20
        vpT = vT * (1+0.61*w)
    else:
        # exact formula
        vpT = vT * (1+(w/mwr)/(1+w))
    return vpT

def wind_components(speed: float, direction: float) -> tuple[float, float]:
    # given a wind speed and a direction in degrees CW of N,
    # return u, v (eastward, northward) components of wind
    if math.isnan(direction):
        return 0., 0.
    direction_rad = np.radians(direction)
    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)
    return u, v

# bulk Richardson number -  computes bulk Ri given data at two heights (z1 and z2)
# vpt(1/2): virtual potential temperature, K
# z(1/2): height, m
# ws(1/2): wind speed, m/s
# wd(1/2): wind direction, degrees E of N
# g [OPT]: local gravity, m/s^2
def bulk_richardson_number(vpt1: float, vpt2: float, z1: float, z2: float, ws1: float, ws2: float, wd1: float, wd2: float, g: float=LOCAL_GRAVITY_DEFAULT) -> float:
    g = LOCAL_GRAVITY
    delta_vpt = vpt2 - vpt1
    delta_z = z2 - z1
    u1, v1 = wind_components(ws1, wd1)
    u2, v2 = wind_components(ws2, wd2)
    delta_u = u2 - u1
    delta_v = v2 - v1
    if delta_u == 0 and delta_v == 0:
        return np.nan
    vpt_avg = (vpt1 + vpt2) / 2
    ri = g * delta_vpt * delta_z / (vpt_avg * (delta_u * delta_u + delta_v * delta_v))
    return ri

# stability classification based on bulk Richardson number
# implementation based on Neumann & Klein (2014): https://doi.org/10.3390/resources3010081, as done in Ahlman-Zhang-Markfort manuscript
def bulk_stability_class(Ri_b: float) -> str:
    if Ri_b < -0.1:
        return 'unstable'
    if -0.1 <= Ri_b < 0.1:
        return 'neutral'
    if 0.1 <= Ri_b < 0.25:
        return 'stable'
    return 'strongly stable'

"""
def cedar_rapids_terrain_class(direction):
    if 300. <= direction <= 330.:
        # northwest sector = complex terrain
        return "complex"
    elif 120. <= direction <= 150.:
        # southwest sector = open terrain
        return "open"
    return "other"
"""
        
class TerrainClassifier:
    def __init__(self, open_terrain: tuple[float, float], complex_terrain: tuple[float, float]):
        self._a, self._b = open_terrain
        self._c, self._d = complex_terrain

    def __eval__(self, direction: float):
        return self.classify(direction)
    
    def classify(self, direction: float):
        if self._c <= direction <= self._d:
            return "complex"
        if self._a <= direction <= self._b:
            return "open"
        if self._a > self._b: # 360-wraparound for open
            if self._a <= direction or self._b >= direction:
                return "open"
        if self._c > self._d: # 360-wraparound for closed
            if self._c <= direction or self._d >= direction:
                return "complex"
        return "other"

