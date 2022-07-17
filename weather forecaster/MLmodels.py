from AMARIA import model1
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


def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        #try: 
        model = SARIMAX(exog, order=order).fit(disp=-1)
    #except:
    #        continue
            
        aic = model.aic
        results.append([order, model.aic])
    #print(results)
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df

def run(b,c):
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
    ps = range(0, 10, 1)
    d = 0
    qs = range(0, 10, 1)

    # Create a list with all possible combination of parameters
    parameters = product(ps, qs)
    parameters_list = list(parameters)

    order_list = []

    for each in parameters_list:
        each = list(each)
        each.insert(1, d)
        each = tuple(each)
        order_list.append(each)
        
    result_d_0 = optimize_ARIMA(order_list, exog = series)

    ps = range(0, 10, 1)
    d = 1
    qs = range(0, 10, 1)

    # Create a list with all possible combination of parameters
    parameters = product(ps, qs)
    parameters_list = list(parameters)

    order_list = []

    for each in parameters_list:
        each = list(each)
        each.insert(1, d)
        each = tuple(each)
        order_list.append(each)
        
    result_d_1 = optimize_ARIMA(order_list, exog = series)

    result_d_1
    result_d_1.head()
    final_result = result_d_0.append(result_d_1)

    
    best_models = final_result.sort_values(by='AIC', ascending=True).reset_index(drop=True).head()
    best_model_params_0 = best_models[best_models.columns[0]][0]
    best_model_params_1 = best_models[best_models.columns[0]][1]
    best_model_0 = SARIMAX(series, order=best_model_params_0).fit()
    print(best_model_0.summary())
    best_model_1 = SARIMAX(series, order=best_model_params_1).fit()
    print(best_model_1.summary())
    best_model_0.plot_diagnostics(figsize=(15,12))
    plt.show()
    best_model_1.plot_diagnostics(figsize=(15,12))
    plt.show()
    fore_l= test_size-1

    forecast = best_model_0.get_prediction(start=training_size, end=training_size+fore_l)
    forec = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    s_forecast = best_model_1.get_prediction(start=training_size, end=training_size+fore_l)
    s_forec = s_forecast.predicted_mean
    s_ci = forecast.conf_int(alpha=0.05)
    error_test=b.loc[test_date[1:].index.tolist()].AverageTemperatureUncertainty
    index_test = test_date[1:].index.tolist()
    test_set = test_series[1:]
    lower_test = test_set-error_test
    upper_test = test_set+error_test
    fig, ax = plt.subplots(figsize=(16,8), dpi=300)
    x0 = b.AverageTemperature.index[0:training_size]
    x1=b.AverageTemperature.index[training_size:training_size+fore_l+1]
    #ax.fill_between(forec, ci['lower Load'], ci['upper Load'])
    plt.plot(x0, b.AverageTemperature[0:training_size],'k', label = 'Average Temperature')

    plt.plot(b.AverageTemperature[training_size:training_size+fore_l], '.k', label = 'Actual')

    forec = pd.DataFrame(forec, columns=['f'], index = x1)
    #forec.f.plot(ax=ax,color = 'Darkorange',label = 'Forecast (d = 2)')
    #ax.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')

    s_forec = pd.DataFrame(s_forec, columns=['f'], index = x1)
    s_forec.f.plot(ax=ax,color = 'firebrick',label = 'Forecast  (2,1,6) model')
    ax.fill_between(x1, s_ci['lower AverageTemperature'], s_ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')


    plt.legend(loc = 'upper left')
    plt.xlim(80,)
    plt.xlabel('Index Datapoint')
    plt.ylabel('Temperature')
    plt.show()

    #plt.plot(forec)
    plt.figure(figsize=(12,12))
    plt.subplot(2,1,1)
    plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
    plt.plot(test_set,marker='.',label="Actual",color='navy')
    plt.plot(forec,marker='d',label="Forecast",color='firebrick')
    plt.xlabel('Index Datapoint')
    plt.ylabel('Temperature')
    #plt.fill_between(x1, s_ci['lower AverageTemperature'], s_ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
    plt.legend()
    plt.subplot(2,1,2)
    #plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
    plt.plot(test_set,marker='.',label="Actual",color='navy')
    plt.plot(s_forec,marker='d',label="Forecast",color='firebrick')
    plt.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
    plt.legend()
    plt.xlabel('Index Datapoint')
    plt.ylabel('Temperature')

    plt.fill_between(np.arange(0,len(test_set),1), lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
    plot_from_data(test_set,test_date,c='navy',label='Actual')
    plot_from_data(forec['f'],test_date,c='firebrick',label='Forecast')
    plt.legend(loc=2)
    plt.show()