import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
def get_timeseries(b, start_year, end_year):
    b_data = b[(b["dt"]>=start_year) & (b["dt"]<=end_year)].reset_index().drop(columns=['index'])
    print(len(b_data))
    return b_data

def plot_timeseries(b_data,start_year,end_year):
    b_data = get_timeseries(b_data,start_year,end_year)
    P = np.linspace(0,len(b_data)-1,5).astype(int)
    plt.plot(b_data.AverageTemperature,marker='.',color='firebrick')
    #plt.xticks(np.arange(0,len(b_data),1)[P],b_data.dt.loc[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')

def plot_from_data(data,time,c='firebrick',with_ticks=True,label=None):
    time = time.tolist()
    data = np.array(data.tolist())
    P = np.linspace(0,len(data)-1,5).astype(int)
    time = np.array(time)
    if label==None:
        plt.plot(data,marker='.',color=c)
    else:
        plt.plot(data,marker='.',color=c,label=label)
    if with_ticks==True:
        plt.xticks(np.arange(0,len(data),1)[P],time[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')