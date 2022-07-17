from mechanize import Missing
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from list_cities import list_cites
from missing_data import data
#from calculations import calc
from extractdates import get_timeseries,plot_from_data,plot_timeseries
from dataplt import plot_data
from AMARIA import model1
from MLmodels import optimize_ARIMA, run
def main():
    #ask user for city
    a = list_cites()
    #fill in missing data
    b = data(a)
    #start_year = input("Start year here:")
    #end_year = input("End Year here:")
    start_year = "1990"
    end_year = "2000"
    c = get_timeseries(b,start_year,end_year)
    d = plot_timeseries(c, start_year, end_year)
    f = plot_data(b)
    #g = model1(c,b)
    run(b,c)
    #calculations
    #f = calc()

main()