
GLWind Codebase and Documentation
=================================

Information
----------

**Authors:** Elliott Walker, Hudson Hart, Chloe Amoroso  
**Creation Date:** 8 July 2024  
**Last Update:** 25 September 2024  

This is a repository of the code and data used in the [Summer 2024 GLWind REU](https://engineering.csuohio.edu/glwind_reu/glwind_reu) program. It was designed both for our own use and to allow others in the future to more easily build upon our work. The majority of this repository is dedicated to the code first used for analysis on a dataset from Cedar Rapids, Iowa for the project *Wind profile characterization based on surface terrain and atmospheric thermal stability*. 

UPDATE (Elliott - 25 September 2024)
--------------------------
Hi everyone, I didn't end up having time to finish the `windprofiles` package, and this repository got messy anyways. For everything important to the [wind profile analysis/characterization project](https://engineering.csuohio.edu/glwind_reu/wind-profile-characterization-based-surface-terrain-and-atmospheric-thermal-stability) - that is, pretty much anything that doesn't have to do with the sensor installation, excepting the unfinished `windprofiles` package, see my new repository [windprofile_characterization](https://github.com/Intergalactyc/windprofile_characterization). It will have most(/all?) of the code actually being used in the analysis, which I'm currently working on in Fall '24. Hopefully this glwind_codebase repo is still useful so that you can see a complete record of all that we did during Summer 2024; for the current state of the analysis though you probably want to look at the new one. Feel free to reach out to me at walker.elliott.j@gmail.com if you're confused by the organization between these or anything else regarding the code!

Repository contents
-------------------

* `/DATA` contains the raw data used for analysis.  
* `/windprofiles` is a package which has various capabilities for use in the wind profile analysis.
* `/analysis` has the current state of the actual data analysis, based on the contents of DATA and using the `windprofiles` package.
* `/original` contains the original code that we used for our analysis, before we began porting it into the windprofiles package. Lots of this may be useful.  
* `/sample_analyses` has some sample data and example analyses of these data.
* `/docs` contains extra documentation files, regarding various aspects of this project.  
* `/assets` contains images and other miscellaneous assets.  

Windprofiles package
--------------------

See `/windprofiles/README.md` for information on package installation and usage.

There's a lot left to be done with this package. For now, the *complete* analysis is only available in `/old_code`, future additions may be made to `/windprofiles` to incorporate the rest of the analysis there.

Some key code is located in `/old_code`. For example, `/old_code/sonic.py` is a strong standalone program for some processing steps of sonic data.
