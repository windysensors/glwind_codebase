from windprofiles.objects import Boom, MetTower
from atmo_calc import TerrainClassifier
from units import Quantity
import os

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
    verbose=True
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
    verbose=True
)

# boom 3, at 20 meters
crBoom3 = Boom.generate(
    crDataPath+'Boom3OneMin.csv',
    height = Quantity(20.,'m'),
    time = 'TIMESTAMP',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    verbose=True
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
    verbose=True
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
    verbose=True
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
    verbose=True
)

# boom 7, at 106 meters
crBoom7 = Boom.generate(
    crDataPath+'Boom7OneMin.csv',
    height = Quantity(106.,'m'),
    time = 'TimeStamp',
    ws = ('MeanVelocity (m/s)','m/s'),
    wd = ('MeanDirection','E-CW'),
    P = ('MeanPressure (mmHg)','mmHg'),
    verbose=True
)

# use the pressure data from boom 1 in boom 2
crBoom2.partial_merge(crBoom1, 'P')

# standard merge of booms 6 and 7
crBoomTop = crBoom6 + crBoom7

# form the met tower by combining all booms
cedarRapidsTower = MetTower(
    name = 'Cedar Rapids, Ohio',
    booms = [crBoom1, crBoom2, crBoom3, crBoom4, crBoom5, crBoom6+crBoom7],
    latitude = 41.9779,
    longitude = 91.6656,
    terrain_class = crTerrainClassifier
)

# data cleaning and basic calculations
cedarRapidsTower.remove_outliers(verbose=True)
cedarRapidsTower.resample(10, verbose=True)
cedarRapidsTower.compute_vpt(verbose=True)
