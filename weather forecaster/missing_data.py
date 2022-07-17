import pandas as pd
from list_cities import list_cites

def data(city):
    city["AverageTemperature"]=city.AverageTemperature.fillna(method='bfill')
    city["AverageTemperatureUncertainty"]=city.AverageTemperature.fillna(method='bfill')
    #print (city)
    return city
    