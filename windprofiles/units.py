### units.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: Quantity object class with conversion functionality

from __future__ import annotations
import numpy as np

pressures_kPa = {
    # linear conversion pair (a,b): <value in kPa> = a * (<value in UNIT> + b)
    'kPa' : (1,0), # kilopascal
    'hPa' : (1/10,0), # hectopascal
    'Pa' : (1/1000,0), # pascal
    'mmHg' : (0.133322,0), # millimeter of mercury
    'bar' : (100,0), # bar
    'mbar' : (10.,0), # millibar
    'psi' : (6.89476,0), # pound force per square inch
    'atm' : (101.325,0) # atmosphere
}

temperatures_K = {
    # linear conversion pair (a,b): <value in K> = a * (<value in UNIT> + b)
    'K' : (1,0), # Kelvin
    'deg C' : (1.,273.15), # degree Celsius
    'C' : (1.,273.15), # degree Celsius, alternative
    'deg F' : (5/9,827.406), # degree Fahrenheit
    'F' : (5/9,827.406) # degree Farenheit, alternative
}

windspeeds_mtps = {
    # linear conversion pair (a,b): <value in m/s> = a * (<value in UNIT> + b)
    'm/s' : (1,0), # meter per second
    'mph' : (0.44704,0) # mile per hour
}

lengths_m = {
    # linear conversion pair (a,b): <value in m> = a * (<value in UNIT> + b)
    'm' : (1,0), # meter
    'km' : (1000,0), # kilometer
    'ft' : (0.3048,0), # foot
    'mi' : (1609.34,0) # mile
}

relhumidity_dec = {
    # linear conversion pair (a,b): <decimal value> = a * (<value in UNIT> + b)
    'dec' : (1,0), # decimal
    'decimal' : (1,0), # decimal, alternative
    'percent' : (0.01,0), # percentage
    '%' : (0.01,0) # percentage, alternative
}

direction_NCW = {
    # linear conversion pair (a,b): <value in NtoE> = a * (<value in UNIT> + b)
    # third value is True to designate this as needing to be conducted mod 360
    'N-CW' : (1,0,True), # degrees from North in clockwise direction
    'N-CC' : (-1,0,True), # degrees from North in counterclockwise direction
    'E-CW' : (1,-90,True), # degrees from East in clockwise direction
    'E-CC' : (-1,-90,True) # degrees from East in counterclockwise direction
}

unit_search = {
    'ws' : windspeeds_mtps,
    'wd' : direction_NCW,
    'P' : pressures_kPa,
    'T' : temperatures_K,
    'z' : lengths_m,
    'rh' : relhumidity_dec
}

defaults = {
    'ws' : 'm/s',
    'wd' : 'N-CW',
    'P' : 'kPa',
    'T' : 'K',
    'z' : 'm',
    'rh' : 'dec'
}

def _find_dimension(unit):
    found = False
    for dimname, dimunits in unit_search.items():
        if unit in dimunits:
            dimension = dimname
            found = True
    if found:
        return dimension
    return False

def convert_value(value, oldUnit, newUnit, *, infer=True, dimension=None):
    if infer:
        dimension = _find_dimension(oldUnit)
        if not dimension:
            raise QuantityError(f'Could not convert {oldUnit} to {newUnit}: unit {oldUnit} not found')
            return value
        if newUnit not in unit_search[dimension].keys():
            raise QuantityError(f'Could not convert {oldUnit} to {newUnit}: invalid or mismatched dimensions')
            return value
    else:
        if dimension not in unit_search.keys():
            raise QuantityError(f'Could not find dimension {dimension}')
    a1, b1, *_ = unit_search[dimension][oldUnit] + (None,)
    a2, b2, isDir, *_ = unit_search[dimension][newUnit] + (None,)
    standardized = a1 * (value + b1)
    if isDir:
        standardized %= 360
    converted = standardized/a2 - b2
    if isDir:
        converted %= 360
    return converted

class Quantity():
    def __init__(self, value: float, unit: str):
        self._value = float(value)
        self._unit = unit
        if not _find_dimension(unit):
            print(f'WARNING: unit {unit} not found')

    def __str__(self):  
        return f'{self._value} {self._unit}'
    
    def __repr__(self):
        return self.__str__()
    
    def __float__(self):
        return self._value
    
    def __add__(self, other: Quantity):
        if '_unit' in dir(other) and other._unit != self._unit:
            raise QuantityError(f'Could not add quantities of {self._unit} and {other._unit}')
        return float.__add__(self._value, other)
    
    def __eq__(self, other: Quantity):
        if '_unit' in dir(other) and '_value' in dir(other):
            if self._unit == other._unit:
                return self._value == other._value
            return self.convert('kPa') == other.convert('kPa')
        return float.__eq__(self._value, other)
    
    def convert(self, newUnit: str) -> Quantity:
        return Quantity(self.in_units_of(newUnit), newUnit)
    
    def in_units_of(self, newUnit: str):
        return convert_value(self._value,self._unit,newUnit)

# NEED TO TEST THE QUANTITY CLASS TO SEE IF FLOAT CASTING IS IMPLICITLY CALLED ON *, ETC
