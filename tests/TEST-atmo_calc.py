### TEST-atmo_calc.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: tests for atmo_calc.py

# ERROR CODES (0 - success):
#   1 - local_gravity
#   2 - saturation_vapor_pressure

import context
import windprofile_analysis.atmo_calc as atmo_calc
from math import isclose

def _UT_local_gravity():
    # test not yet implemented
    return False

def _UT_saturation_vapor_pressure():
    # test not yet implemented
    return False

def RUN_TESTS():
    if _UT_local_gravity():
        return 1
    if _UT_saturation_vapor_pressure():
        return 2 
    return 0

if __name__ == '__main__':
    result = RUN_TESTS()
    assert result == 0, f"Failure in module 'atmo_calc' - status code {result}"
