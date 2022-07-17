#ik i spelt the .py file wrong idc
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import folium
import imageio
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import imageio
import mapclassify as mc
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf
from mechanize import Missing
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from list_cities import list_cites
from missing_data import data
#from calculations import calc
from extractdates import get_timeseries,plot_from_data,plot_timeseries
from dataplt import plot_data

def model1(c,b):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(c.AverageTemperature, ax=ax1,color ='firebrick')
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(c.AverageTemperature, ax=ax2,color='firebrick')
    temp = get_timeseries(b,"1992","2013")
    N = len(temp.AverageTemperature)
    split = 0.95
    training_size = round(split*N)
    test_size = round((1-split)*N)
    series = temp.AverageTemperature[:training_size]
    date = temp.dt[:training_size]
    test_series = temp.AverageTemperature[len(date)-1:len(temp)]
    test_date = temp.dt[len(date)-1:len(temp)]
    plot_from_data(series,date,label='Training Set')
    plot_from_data(test_series,test_date,'navy',with_ticks=False,label='Test Set')
    plt.legend()
    plt.show()
    return series

