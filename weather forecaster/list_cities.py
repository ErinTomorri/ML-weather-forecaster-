import csv, os
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#this will print out all the cities in the world to determine the forecast
def list_cites():
    cities = []
    data = pd.read_csv("C:/Users/Erin Tomorri/Desktop/weather forecaster/Data/GlobalLandTemperaturesByMajorCity.csv")

    #for num in range(len(data[data['City']])):
    total_city_names=data['City']
    df2 = total_city_names.drop_duplicates(keep = "first")
    df2 = df2.values.tolist()
    #test = input("Do you want to see the list of cities (Y/N): ")
    #test = test.upper()
    test = "N"
    if test == "Y":
        print(df2)
    #name = input("What city do you want: ")
    name = "Chicago"
    if name not in df2:
        print("Your city is unavaliable, as data is not sufficient")
    else:
        city_data = data[data['City']==name]
        return city_data


