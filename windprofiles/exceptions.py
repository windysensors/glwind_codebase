### exceptions.py ###
# author: Elliott Walker
# last update: 11 July 2024
# description: Custom exception types

# NameNotFound exception for improper data column / object access in Boom or MetTower
class NameNotFound(Exception):
    pass

# QuantityError exception for general units.Quantity errors
class QuantityError(Exception):
    pass

# IncompatibilityError exception for general incompatibility-related errors
class IncompatibilityError(Exception):
    pass

# FilehandlingError exception for errors in saving/loading Boom or MetTower objects
class FilehandlingError(Exception):
    pass

# AvailabilityError exception for trying to perform a calculation with a dataset lacking the necessary data
class AvailabilityError(Exception):
    pass

# ModeNotFound exception for improper plot mode name
class ModeNotFound(Exception):
    pass
