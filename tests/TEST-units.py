### TEST-atmo_calc.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: tests for units.py

# ERROR CODES (0 - success):
#   1 - Quantity operators
#   2 - Quantity conversion
#   3 - Quantity getValue

import context
import traceback
import windprofiles.units as units

def _UT_Quantity_operators():
    try:
        q1 = 
    except:
        traceback.print_exc()
        return True
    
def _UT_Quantity_conversion():
    try:
        
    except:
        traceback.print_exc()
        return True
    
def _UT_Quantity_getValue():
    try:
        
    except:
        traceback.print_exc()
        return True

def RUN_TESTS():
    if _UT_Quantity_operators():
        return 1
    if _UT_Quantity_conversion():
        return 2 
    if _UT_Quantity_getValue():
        return 3
    return 0
