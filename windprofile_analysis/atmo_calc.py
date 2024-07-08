### atmo_calc.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: functions for atmosphere-related calculations

import numpy as np

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

def wind_components(speed: float, direction: float) -> tuple(float, float):
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

# terrain classification based on wind direction for Cedar Rapids, Iowa met tower
# implementation based on work in Ahlman-Zhang-Markfort manuscript
def cedar_rapids_terrain_class(direction):
    if 300. <= direction <= 330.:
        # northwest sector = complex terrain
        return "complex"
    elif 120. <= direction <= 150.:
        # southwest sector = open terrain
        return "open"
    return "other"

# least squares linear fit - fits data in (x,y) pairs to a relationship y = A + B*x
# xvals and yvals should each be iterables of the same length
# outputs pair A,B
def ls_linear_fit(xvals, yvals):
    if len(yvals) == 0 or len(xvals) == 0:
        return 0,0
    xvals = list(xvals)
    yvals = list(yvals)
    if len(yvals) != len(xvals):
        throw("Lists must be of equal size")
    for x, y in zip(xvals, yvals):
        if math.isnan(y):
            xvals.remove(x)
            yvals.remove(y)
    n = len(xvals)
    sum_x = sum(xvals)
    sum_x2 = sum(x*x for x in xvals)
    sum_xy = sum(xvals[i]*yvals[i] for i in range(n))
    sum_y = sum(yvals)
    det = n * sum_x2 - sum_x * sum_x
    A = (sum_y * sum_x2 - sum_x * sum_xy)/det
    B = (n * sum_xy - sum_x * sum_y)/det
    return A, B

# power law fit - fits data in (x,y) pairs to a relationship y = A*x**B
# xvals and yvals should each be iterables of the same length
# pass argument both=True to obtain both A and B
#   otherwise only the exponent B will be returned (wind shear exponent)
# outputs either just B or pair A,B based on "both"
def power_fit(xvals, yvals, both=False):
    xconsider = []
    yconsider = []
    for x,y in zip(xvals, yvals):
        if not (math.isnan(x) or math.isnan(y)):
            xconsider.append(x)
            yconsider.append(y)
    lnA, B = ls_linear_fit(np.log(xconsider),np.log(yconsider))
    if both:
        return np.exp(lnA), B
    return B
