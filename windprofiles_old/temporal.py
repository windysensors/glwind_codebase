### temporal.py ###
# author: Elliott Walker
# last update: 11 July 2024
# description: Functionality for handling temporal variation

months = {1 : 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

# Selector object class
# For classification of times based on intervals they may fall within
class Selector:
    def __init__(self):
        pass
    
    def select(self, time, choice):
        # returns True if <time> meets the conditions of being categorized as <choice>, False otherwise
        pass

    def classify(self, times: list):
        # categorically classifies each time in a list of times
        pass

    @staticmethod
    def month():
        # generate a Selector object that will select a single month
        return Selector()

# NOTHING REALLY HERE YET, JUST STARTING TO SET UP THE FRAMEWORK
