from mechanize import Missing
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from list_cities import list_cites
from missing_data import data
#from calculations import calc
from extractdates import get_timeseries,plot_from_data,plot_timeseries
#this is more for testing than anything, ignore
def plot_data(b):
    plt.figure(figsize=(20,20))
    plt.suptitle('Plotting 4 decades',fontsize=40,color='firebrick')
    plt.subplot(2,2,1)
    plt.title('Starting year: 1800, Ending Year: 1810',fontsize=15)
    plot_timeseries(b,"1800","1810")
    plt.subplot(2,2,2)
    plt.title('Starting year: 1900, Ending Year: 1910',fontsize=15)
    plot_timeseries(b,"1900","1910")
    plt.subplot(2,2,3)
    plt.title('Starting year: 1950, Ending Year: 1960',fontsize=15)
    plot_timeseries(b,"1900","1910")
    plt.subplot(2,2,4)
    plt.title('Starting year: 2000, Ending Year: 2010',fontsize=15)
    plot_timeseries(b,"1900","1910")
    plt.tight_layout()