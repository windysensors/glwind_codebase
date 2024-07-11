### iowa_analysis.py ###
# author: Elliott Walker
# last update: 11 July 2024
# description: Uses windprofiles package to perform sample analysis on the Cedar Rapids, Iowa dataset

from windprofiles.objects import Boom, MetTower
from atmo_calc import TerrainClassifier
from units import Quantity
import os

VERBOSE = False

# terrain classification based on wind direction for Cedar Rapids, Iowa met tower
# boundaries based on work in Ahlman-Zhang-Markfort manuscript
crTerrainClassifier = TerrainClassifier(
    open_terrain = (300.,330.),
    complex_terrain = (120.,150.)
)

crDataPath = os.path.dirname(os.path.abspath(__file__)) + '/../DATA/CedarRapids/'

# boom 1, at 6 meters
crBoom1 = Boom.generate(
    crDataPath+'Boom1OneMin.csv',
    height = Quantity(6.,'m'),
    time = 'TimeStamp',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    T = ('MeanTemperature (C )','deg C'),
    P = ('MeanPressure (mmHg)','mmHg'),
    verbose=VERBOSE
)

# boom 2, at 10 meters
crBoom2 = Boom.generate(
    crDataPath+'Boom2OneMin.csv',
    height = Quantity(10.,'m'),
    time = 'TIMESTAMP',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    rh = ('MeanRH (%)','percent'),
    T = ('MeanTemperature (C )','deg C'),
    verbose=VERBOSE
)

# boom 3, at 20 meters
crBoom3 = Boom.generate(
    crDataPath+'Boom3OneMin.csv',
    height = Quantity(20.,'m'),
    time = 'TIMESTAMP',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    verbose=VERBOSE
)

# boom 4, at 32 meters
crBoom4 = Boom.generate(
    crDataPath+'Boom4OneMin.csv',
    height = Quantity(32.,'m'),
    time = 'TimeStamp',
    ws = ('MeanVelocity','m/s'),
    wd = ('MeanDirection','E-CW'),
    rh = ('MeanRH','percent'),
    T = ('MeanTemperature','deg C'),
    verbose=VERBOSE
)

# boom 5, at 80 meters
crBoom5 = Boom.generate(
    crDataPath+'Boom5OneMin.csv',
    height = Quantity(80.,'m'),
    time = 'TimeStamp',
    ws = ('MeanVelocity','m/s'),
    wd = ('MeanDirection','E-CW'),
    rh = ('MeanRH','percent'),
    T = ('MeanTemperature','deg C'),
    verbose=VERBOSE
)

# boom 6, at 106 meters
crBoom6 = Boom.generate(
    crDataPath+'Boom6OneMin.csv',
    height = Quantity(106.,'m'),
    time = 'TIMESTAMP',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('Mean Direction','E-CW'),
    rh = ('MeanRH (%)','percent'),
    T = ('MeanTemperature (C )','deg C'),
    verbose=VERBOSE
)

# boom 7, at 106 meters
crBoom7 = Boom.generate(
    crDataPath+'Boom7OneMin.csv',
    height = Quantity(106.,'m'),
    time = 'TimeStamp',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    P = ('MeanPressure (mmHg)','mmHg'),
    verbose = VERBOSE
)

# use the pressure data from boom 1 in boom 2
crBoom2.partial_merge(crBoom1, 'P')

# standard merge of booms 6 and 7
crBoomTop = crBoom6 + crBoom7

# form the met tower by combining all booms
cedarRapidsTower = MetTower(
    name = 'Cedar Rapids, Ohio',
    booms = {
        '6m' : crBoom1,
        '10m' : crBoom2,
        '20m' : crBoom3,
        '32m' : crBoom4,
        '80m' : crBoom5,
        '106m' : crBoomTop
    },
    latitude = 41.9779,
    longitude = 91.6656,
    terrain_class = crTerrainClassifier
)

# data cleaning and basic calculations
cedarRapidsTower.remove_outliers(n_samples=30, sigma=5, verbose=VERBOSE)
cedarRapidsTower.resample(n_samples=10, verbose=VERBOSE)
cedarRapidsTower.compute_vpt(verbose=VERBOSE)
cedarRapidsTower.associate_canonical_time(which_booms=['10m','106m'], verbose=VERBOSE)
cedarRapidsTower.compute_ri('10m','106m', verbose=VERBOSE)
cedarRapidsTower.save_windrose('10m', 'PLOTS/10mRose.png', mode='speed', verbose=VERBOSE)
cedarRapidsTower.save_windrose('10m', 'PLOTS/riRose.png', mode='ri', verbose=VERBOSE)
cedarRapidsTower.save_stabilities('PLOTS/stabilitiesBar.png', verbose=VERBOSE)
