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

To use the contents of `/windprofiles`, it must be installed. Start from the codebase directory `glwind_codebase/`.  
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


* `atmo_calc.py` has various helpful functions for atmosphere-related calculations.  
* `stat_calc.py` has some statistical functions, including functions for angular averaging and multiple forms of least squares regression.
* `objects.py` defines Boom and MetTower objects.  
* `exceptions.py` defines a few custom exception types.
* `setup.py` and `__init__.py` can be ignored for basic use. If making updates to the package, they must be changed for versioning: most importantly, newly created python files should be listed in the py_modules variable of `setup.py` and imported in `__init__.py`.  

### Basic use:

