### Getting started:

To use the contents of `/windprofiles`, it must be installed. Run `pip install -e windprofiles` from the codebase directory, then you will be able to access its contents across your system (e.g. `import windprofiles.atmo_calc as atmo_calc`). If the package is modified, make sure to reinstall in order to keep the pip package up-to-date.

### Package contents:

* `setup.py` and `__init__.py` can be ignored for basic use. If making updates to the package, `setup.py` must be changed for versioning.  
* `atmo_calc.py` has various helpful functions for atmosphere-related calculations.  
* `objects.py` defines Boom and MetTower objects.  
