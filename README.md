
GLWind Codebase and Documentation
=================================

Information
----------

**Authors:** Elliott Walker, Hudson Hart, Chloe Amoroso  
**Creation Date:** 8 July 2024  
**Last Update:** 9 July 2024
This is a repository of the code and data used in the [Summer 2024 GLWind REU](https://engineering.csuohio.edu/glwind_reu/glwind_reu) program. It was designed both for our own use and to allow others in the future to more easily build upon our work. The majority of this repository is dedicated to the code first used for analysis on a dataset from Cedar Rapids, Iowa for the project *Wind profile characterization based on surface terrain and atmospheric thermal stability*.  

Repository contents
-------------------

* `/DATA` contains the raw data used for analysis.  
* `/windprofiles` is a package which has various capabilities for use in the wind profile analysis.
* `/analysis` has the current state of the actual data analysis, based on the contents of DATA and using the `windprofiles` package.
* `/tests` contains test files for validating Python code from `windprofiles'. So far, there are no real functioning tests, just a simplistic framework that can be built upon.
* `/docs` contains various documentation files, regarding all aspects of this project.  
* `/assets` contains images and other miscellaneous assets.  

Windprofiles package
--------------------

See `/windprofiles/README.md` for information on package installation and usage.
