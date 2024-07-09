windprofiles package
====================

Information
----------

**Authors:** Elliott Walker, Hudson Hart, Chloe Amoroso  
**Creation Date:** 8 July 2024  
**Last Update:** 9 July 2024  
This package contains code for handling and analyzing certain types of meteorological tower (met tower) data.


Getting started
---------------


### Installation:

To use the contents of `windprofiles`, the package must be installed. Start from the parent directory (the directory containing `windprofiles/`, e.g. `glwind_codebase/`)
1. From the codebase directory, run `pip install -e windprofiles`.  
    * This installs the package in editing mode, so changes made should be reflected in new imports.  
2. Add the codebase directory to your PYTHONPATH environment variable.  
    * On Windows: `set PYTHONPATH=%CD%;%PATH%`  
    * On Linux/MacOS: `export PYTHONPATH=$(pwd):$PYTHONPATH`  

You should now be able to access the contents of the package systemwide. If the package is modified, make sure to reinstall in order to keep the pip package up-to-date.  

> This will only add the directory to PYTHONPATH for your current shell. To make this change permanent, do the following so that a new shell will have the variable already set:
* On Windows: `setx PYTHONPATH %CD%;%PATH%`
* On Linux/MacOS:
    * First, in the codebase directory call `pwd` to see its full absolute path, e.g. /home/user/directory/glwind_codebase
    * Now, add `export PYTHONPATH=``<absolute_path_here``>:$PYTHONPATH` to the end of your .bashrc file (a hidden file; may be located in the home directory ~/.bashrc, or possibly in /etc/.bashrc).

### Package contents:

* `atmo_calc.py` (windprofiles.atmo_calc module) has various helpful functions for atmosphere-related calculations.  
* `stat_calc.py` (windprofiles.stat_calc module) has some statistical functions, including functions for angular averaging and multiple forms of least squares regression.
* `objects.py` (windprofiles.objects module) defines Boom and MetTower objects.  
* `exceptions.py` (windprofiles.exceptions module) defines a few custom exception types.
* `setup.py` and `__init__.py` can be ignored for basic use. If making updates to the package, they must be changed for versioning: most importantly, newly created python files should be listed in the py_modules variable of `setup.py` and imported in `__init__.py`.  


### Basic use:

Import the package functionality into your own code, typically by module. For example:  

```python
    import windprofiles.stat_calc as stat_calc
    from windprofiles.objects import Boom
```

The core parts of this package are the `Boom` and `MetTower` classes. A Boom contains time-series data at a single height, and is at its core a wrapper for a pandas DataFrame with a well-defined structure & various additional methods. A Boom can be loaded from an existing pandas DataFrame, or generated from a CSV file by providing the column labels and units corresponding to each variable of interest (as in the example below). A MetTower is essentially a collection of Booms, which allows for bulk processing. For example:

```python
    from windprofiles.units import Quantity
    from windprofiles.objects import Boom, MetTower
    
    file1 = 'path/to/file1.csv'
    file2 = 'path/to/file2.csv'

    # Load the first boom
    boom1 = Boom.generate(
        filename = file1,               # path to csv file to generate Boom from
        height = Quantity(30., 'ft'),   # Quantity describing height above ground
        time = 'Timestamp',             # column label (in .csv file) for time stamp
        ws = ('Wind Speed', 'mph'),     # label and units for wind speed
        wd = ('Wind Direction', 'N-CW') # label and reference direction for wind direction
    )

    # Load the second boom
    boom2 = Boom.generate(
        filename = file2,
        height = 40.,                   # passing a float rather than a Quantity defaults to being interpreted as meters
        time = 'TIME',
        ws = ('Windspeed', 'm/s'),
        wd = ('Wind direction', 'E-CC'),
        rh = ('Relative humidity', 'percent'),  # label and units for relative humidity (percent vs dec[imal])
        T = ('Temperature', 'deg C'),           # label and units for temperature
        P = ('Pressure', 'hPa'),                # label and units for pressure
    )

    # Combine the booms into a MetTower
    meteorologicalTower = MetTower(
        name = 'Sample Meteorological Tower',
        booms = {
            'Lower boom' : boom1,
            'Upper boom' : boom2
        },
        latitude = 31.50,
        longitude = 90.36
    )

    # Because the MetTower instantiation copied the Boom objects, we can delete the originals to free memory
    del boom1, boom2

    .
    .
    .

```

### On units:

`units.py` has some basic unit handling functionality, primarily based on a Quantity object class. Quantities can be converted between different units of pressure, temperature, and speed (as well as from percent<->decimal [for relative humidity] and between reference directions [for wind direction]). See supported_units.txt for a comprehensive list of supported units.  
At least for now, all data is stored internally in a common unit system (see `windprofiles.units.defaults`):  
* Pressure in kPa  
* Temperature in K  
* Wind speeds in m/s  
* Relative humidities as decimals  
* Heights in m  
* Wind directions in N-CW  
    - This means "degrees clockwise from North".  
    - Alternatives include e.g. N-CC ("degrees counterclockwise from North"), E-CW ("degrees clockwise from East")  

The primary use of the conversion system, besides being available for user use, is to provide a framework for converting file/DataFrame data on Boom generation into these common units.  


Objects
--------

