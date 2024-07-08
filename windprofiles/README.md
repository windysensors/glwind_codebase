### Getting started:

To use the contents of `/windprofiles`, it must be installed. Start from the codebase directory `glwind_codebase/`.
1. From the codebase directory, run `pip install -e windprofiles`.
    * This installs the package in editing mode, so changes made should be reflected in new imports.
2. Add the codebase directory to your PYTHONPATH environment variable.  
    * On Windows: `set PYTHONPATH=%CD%;%PATH%`
    * On Linux/MacOS: `export PYTHONPATH=$(pwd):$PYTHONPATH`  
You should now be able to access the contents of the package systemwide (e.g. `import windprofiles.atmo_calc as atmo_calc`; `from windprofiles.objects import Boom`). If the package is modified, make sure to reinstall in order to keep the pip package up-to-date.

### Package contents:

* `setup.py` and `__init__.py` can be ignored for basic use. If making updates to the package, `setup.py` must be changed for versioning.  
* `atmo_calc.py` has various helpful functions for atmosphere-related calculations.  
* `objects.py` defines Boom and MetTower objects.  
