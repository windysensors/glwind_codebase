### units.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: Quantity object class with conversion functionality

import numpy as np

pressures_kPa = {
    # linear conversion pair (a,b): <value in kPa> = a * (<value in UNIT> + b)
    "kPa" : (1,0),
    "hPa" : (1/10,0),
    "Pa" : (1/1000,0),
    "mmHg" : (0.133322,0),
    "bar" : (100,0),
    "mbar" : (10.,0),
    "psi" : (6.89476,0),
    "atm" : (101.325,0)
}

temperatures_K = {
    # linear conversion pair (a,b): <value in K> = a * (<value in UNIT> + b)
    "K" : (1,0),
    "deg C" : (1.,273.15),
    "deg F" : (5/9.,827.406)
}

windspeeds_mtps = {
    # linear conversion pair (a,b): <value in m/s> = a * (<value in UNIT> + b)
    "m/s" : (1,0),
    "mph" : (0.44704,0)
}

lengths_m = {
    # linear conversion pair (a,b): <value in m> = a * (<value in UNIT> + b)
    "m" : (1,0),
    "km" : (1000,0),
    "ft" : (0.3048,0),
    "mi" : (1609.34,0)
}

unit_search = {
    "P" : pressures_kPa,
    "T" : temperatures_K,
    "ws" : windspeeds_mtps,
    "z" : lengths_m
}

class QuantityError(Exception):
    pass

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
            raise QuantityError(f"Could not convert {oldUnit} to {newUnit}: unit {oldUnit} not found")
            return value
        if newUnit not in unit_search[dimension].keys():
            raise QuantityError(f"Could not convert {oldUnit} to {newUnit}: invalid or mismatched dimensions")
            return value
    else:
        if dimension not in unit_search.keys():
            raise QuantityError(f"Could not find dimension {dimension}")
    a1, b1 = unit_search[dimension][oldUnit]
    a2, b2 = unit_search[dimension][newUnit]
    standardized = a1 * (value + b1)
    converted = standardized/a2 - b2
    return converted

class Quantity(float):
    def __new__(cls, value, unit):
        return float.__new__(cls, value)
    def __init__(self, value, unit):
        self._value = float(value)
        self._unit = unit
        if not _find_dimension(unit):
            print(f"WARNING: unit {unit} not found")
    def __str__(self):  
        return f"{self._value} {self._unit}"
    def __repr__(self):
        return self.__str__()
    def __float__(self):
        return self._value
    def __add__(self, other):
        if "_unit" in dir(other) and other._unit != self._unit:
            raise QuantityError(f"Could not add quantities of {self._unit} and {other._unit}")
        return float.__add__(self._value, other)
    def __eq__(self, other):
        if "_unit" in dir(other) and "_value" in dir(other):
            if self._unit == other._unit:
                return self._value == other._value
            return (convert(self, "kPa") == convert(other, "kPa"))
        return float.__eq__(self._value, other)
    def convert(self, newUnit):
        dimension = _find_dimension(self._unit)
        if not dimension:
            raise QuantityError(f"Could not convert {self._unit} to {newUnit}: unit {self._unit} not found")
            return self
        if newUnit not in unit_search[dimension].keys():
            raise QuantityError(f"Could not convert {self._unit} to {newUnit}: invalid or mismatched dimensions")
            return self
        a1, b1 = unit_search[dimension][self._unit]
        a2, b2 = unit_search[dimension][newUnit]
        standardized = a1 * (self._value + b1)
        converted = standardized/a2 - b2
        return Quantity(converted, newUnit)
