#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:49:10 2020

@author: sablelevy
"""

# Import libraries
import sys, os
#os.chdir('/Users/sablelevy/GitHub/fitbit_analyzer')
import fitbit
import gather_keys_oauth2 as Oauth2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import fitbit
#from datetime import timedelta
import csv
from collections import deque


# Function to get your historic Fitbit data from the API.

def get_initial_max_data(authd_client, file_name='Data/Data_master.csv'):
    today = datetime.datetime.today()
    yesterday = today - datetime.timedelta(days=1)
    three_yrs_ago = today - datetime.timedelta(days=1095)
    three_yrs_ago_str = datetime.datetime.strftime(three_yrs_ago,"%Y-%m-%d")
    yesterday_str = datetime.datetime.strftime(yesterday,"%Y-%m-%d")
    #For some reason the max for certain calls is today minus 694 days 
    days_minus_x = today - datetime.timedelta(days=694) #694
    days_minus_x_str = datetime.datetime.strftime(days_minus_x,"%Y-%m-%d")
    #active cals is different:
    days_minus_y = today - datetime.timedelta(days=100) 
    days_minus_y_str = datetime.datetime.strftime(days_minus_y,"%Y-%m-%d")

    steps = authd_client.time_series('activities/steps', base_date= three_yrs_ago_str, end_date=yesterday_str)
    cals = authd_client.time_series('activities/calories', base_date= three_yrs_ago_str, end_date=yesterday_str)
    dist = authd_client.time_series('activities/distance', base_date= three_yrs_ago_str, end_date=yesterday_str)
    floors = authd_client.time_series('activities/floors', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sedant = authd_client.time_series('activities/minutesSedentary', base_date= days_minus_x_str, end_date=yesterday_str)
    elevation = authd_client.time_series('activities/elevation', period='max')
    active_light = authd_client.time_series('activities/minutesLightlyActive',base_date= days_minus_x_str, end_date=yesterday_str)
    active_fair = authd_client.time_series('activities/minutesFairlyActive',base_date= days_minus_x_str, end_date=yesterday_str)
    active_very = authd_client.time_series('activities/minutesVeryActive', base_date= days_minus_x_str, end_date=yesterday_str)
    active_cals = authd_client.time_series('activities/activityCalories', base_date= days_minus_y_str, end_date=yesterday_str)
    sleep_start = authd_client.time_series('sleep/startTime', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_timeInBed = authd_client.time_series('sleep/timeInBed', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_minutesAsleep = authd_client.time_series('sleep/minutesAsleep', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_awakeningsCount = authd_client.time_series('sleep/awakeningsCount', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_minutesAwake = authd_client.time_series('sleep/minutesAwake', base_date= three_yrs_ago_str, end_date=yesterday_str)
  #  sleep_minutesToFallAsleep = authd_client.time_series('sleep/minutesToFallAsleep', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_minutesAfterWakeup = authd_client.time_series('sleep/minutesAfterWakeup', base_date= three_yrs_ago_str, end_date=yesterday_str)
    sleep_efficiency = authd_client.time_series('sleep/efficiency', base_date= three_yrs_ago_str, end_date=yesterday_str)

    body_weight = authd_client.time_series(resource='body/weight', base_date= three_yrs_ago_str, end_date=yesterday_str)
    body_bmi = authd_client.time_series(resource='body/bmi', base_date= three_yrs_ago_str, end_date=yesterday_str)

    
    df = pd.DataFrame()

    df['Date']=pd.DataFrame(steps['activities-steps'])['dateTime']#.astype(datetime)
    df['calories'] = pd.DataFrame(cals['activities-calories'])['value'].astype(int)
    df['steps'] = pd.DataFrame(steps['activities-steps'])['value'].astype(int)
    df['dist'] = pd.DataFrame(dist['activities-distance'])['value'].astype(float)
    df['floors'] = pd.DataFrame(floors['activities-floors'])['value'].astype(int)
    df['elevation'] = pd.DataFrame(elevation['activities-elevation'])['value'].astype(float)
    df['weight']=pd.DataFrame(body_weight['body-weight'])['value'].astype(float)
    df['bmi']=pd.DataFrame(body_bmi['body-bmi'])['value'].astype(float)
    
    df['sleep_start'] = pd.DataFrame(sleep_start['sleep-startTime'])['value']
    df['sleep_timeInBed'] = pd.DataFrame(sleep_timeInBed['sleep-timeInBed'])['value'].astype(int)
    df['sleep_minutesAsleep'] = pd.DataFrame(sleep_minutesAsleep['sleep-minutesAsleep'])['value'].astype(int)
    df['sleep_awakeningsCount'] = pd.DataFrame(sleep_awakeningsCount['sleep-awakeningsCount'])['value'].astype(int)
    df['sleep_minutesAwake'] = pd.DataFrame(sleep_minutesAwake['sleep-minutesAwake'])['value'].astype(int)
 #   df['sleep_minutesToFallAsleep'] = pd.DataFrame(sleep_minutesToFallAsleep['sleep-minutesToFallAsleep'])['value'].astype(int)
    df['sleep_minutesAfterWakeup'] = pd.DataFrame(sleep_minutesAfterWakeup['sleep-minutesAfterWakeup'])['value'].astype(int)
    df['sleep_efficiency'] = pd.DataFrame(sleep_efficiency['sleep-efficiency'])['value'].astype(int)
   
    df_cals=pd.DataFrame()
    df_cals['Date']= pd.DataFrame(active_cals['activities-activityCalories'])['dateTime']
    df_cals['active_cals'] = pd.DataFrame(active_cals['activities-activityCalories'])['value'].astype(int)
    
    df_temp=pd.DataFrame()
    df_temp['sedant']= pd.DataFrame(sedant['activities-minutesSedentary'])['value'].astype(int)
    df_temp['Date']= pd.DataFrame(sedant['activities-minutesSedentary'])['dateTime']
    df_temp['active_light'] = pd.DataFrame(active_light['activities-minutesLightlyActive'])['value'].astype(int)
    df_temp['active_fair'] = pd.DataFrame(active_fair['activities-minutesFairlyActive'])['value'].astype(int)
    df_temp['active_very'] = pd.DataFrame(active_very['activities-minutesVeryActive'])['value'].astype(int)


    df_merged=pd.merge(df_temp, df_cals, how='outer')

    df_master = pd.merge(df, df_merged, how='outer')
    df_master.to_csv(file_name, header=True,index = False)

    return df_master
    
#N=get_initial_max_data(auth2_client)
    

# Function to get your latest Fitbit data from the API.
def get_latest_fitbit_data(authd_client, last_log_date, yesterday):
    
    last_log_date_str = datetime.datetime.strftime(last_log_date,"%Y-%m-%d")
    yesterday_str = datetime.datetime.strftime(yesterday,"%Y-%m-%d")
    
  
    # Retrieve time series data by accessing Fitbit API. 
    # Note that the last logged date is the first date because data only upto yesterday gets written into csv.
    # Today is not over, hence incomplete data from today is not logged.
    
    body_weight = authd_client.time_series(resource='body/weight', base_date= last_log_date_str, end_date=yesterday_str)
    body_bmi = authd_client.time_series(resource='body/bmi', base_date= last_log_date_str, end_date=yesterday_str)

    steps = authd_client.time_series('activities/steps', base_date= last_log_date_str ,end_date=yesterday_str)
    cals = authd_client.time_series('activities/calories', base_date= last_log_date_str ,end_date=yesterday_str)
    dist = authd_client.time_series('activities/distance', base_date= last_log_date_str ,end_date=yesterday_str)
    floors = authd_client.time_series('activities/floors', base_date= last_log_date_str ,end_date=yesterday_str)
    sedant = authd_client.time_series('activities/minutesSedentary', base_date= last_log_date_str ,end_date=yesterday_str)
    elevation = authd_client.time_series('activities/elevation', base_date= last_log_date_str ,end_date=yesterday_str)
    active_light = authd_client.time_series('activities/minutesLightlyActive', base_date= last_log_date_str ,end_date=yesterday_str)
    active_fair = authd_client.time_series('activities/minutesFairlyActive', base_date= last_log_date_str ,end_date=yesterday_str)
    active_very = authd_client.time_series('activities/minutesVeryActive', base_date= last_log_date_str ,end_date=yesterday_str)
    active_cals = authd_client.time_series('activities/activityCalories', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_start = authd_client.time_series('sleep/startTime', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_timeInBed = authd_client.time_series('sleep/timeInBed', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_minutesAsleep = authd_client.time_series('sleep/minutesAsleep', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_awakeningsCount = authd_client.time_series('sleep/awakeningsCount', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_minutesAwake = authd_client.time_series('sleep/minutesAwake', base_date= last_log_date_str ,end_date=yesterday_str)
#    sleep_minutesToFallAsleep = authd_client.time_series('sleep/minutesToFallAsleep', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_minutesAfterWakeup = authd_client.time_series('sleep/minutesAfterWakeup', base_date= last_log_date_str ,end_date=yesterday_str)
    sleep_efficiency = authd_client.time_series('sleep/efficiency', base_date= last_log_date_str ,end_date=yesterday_str)

    df = pd.DataFrame()
    num_days = yesterday - last_log_date + datetime.timedelta(1)
    date_list = [last_log_date.date() + datetime.timedelta(days=x) for x in range(0, num_days.days)]
    df['date'] = date_list
    df['calories'] = pd.DataFrame(cals['activities-calories'])['value'].astype(int)
    df['steps'] = pd.DataFrame(steps['activities-steps'])['value'].astype(int)
    df['dist'] = pd.DataFrame(dist['activities-distance'])['value'].astype(float)
    df['floors'] = pd.DataFrame(floors['activities-floors'])['value'].astype(int)
    df['elevation'] = pd.DataFrame(elevation['activities-elevation'])['value'].astype(float)
    df['weight']=pd.DataFrame(body_weight['body-weight'])['value'].astype(float)
    df['bmi']=pd.DataFrame(body_bmi['body-bmi'])['value'].astype(float)
    
    df['sleep_start'] = pd.DataFrame(sleep_start['sleep-startTime'])['value']
    df['sleep_timeInBed'] = pd.DataFrame(sleep_timeInBed['sleep-timeInBed'])['value'].astype(int)
    df['sleep_minutesAsleep'] = pd.DataFrame(sleep_minutesAsleep['sleep-minutesAsleep'])['value'].astype(int)
    df['sleep_awakeningsCount'] = pd.DataFrame(sleep_awakeningsCount['sleep-awakeningsCount'])['value'].astype(int)
    df['sleep_minutesAwake'] = pd.DataFrame(sleep_minutesAwake['sleep-minutesAwake'])['value'].astype(int)
#    df['sleep_minutesToFallAsleep'] = pd.DataFrame(sleep_minutesToFallAsleep['sleep-minutesToFallAsleep'])['value'].astype(int)
    df['sleep_minutesAfterWakeup'] = pd.DataFrame(sleep_minutesAfterWakeup['sleep-minutesAfterWakeup'])['value'].astype(int)
    df['sleep_efficiency'] = pd.DataFrame(sleep_efficiency['sleep-efficiency'])['value'].astype(int)
    
    df['sedant'] = pd.DataFrame(sedant['activities-minutesSedentary'])['value'].astype(int)
    df['active_light'] = pd.DataFrame(active_light['activities-minutesLightlyActive'])['value'].astype(int)
    df['active_fair'] = pd.DataFrame(active_fair['activities-minutesFairlyActive'])['value'].astype(int)
    df['active_very'] = pd.DataFrame(active_very['activities-minutesVeryActive'])['value'].astype(int)
    df['active_cals'] = pd.DataFrame(active_cals['activities-activityCalories'])['value'].astype(int)
    
    return df

def write_fitbit_data_to_csv(df,data_filename,log_filename):
    # Write the data frame to csv.
    with open(data_filename, 'a') as f:
        df.to_csv(f, header=False,index = False)

    # Log current date as the date last logged.
    today = datetime.datetime.today()
    today_str = datetime.datetime.strftime(today,"%Y-%m-%d")#.encode() #changes to bytes-like object

    with open(log_filename,'w',newline='') as csvfile: #changed to 'w' from 'ab'
        csvwrite = csv.writer(csvfile, delimiter=',')
        csvwrite.writerow([today_str])


def obtain_write_new_data(auth_client):
    
    authd2_client = auth_client


    # Read log file to find out the last day Fitbit data was logged
    with open('Data/Last_log_date.csv') as csvfile:
        temp = deque(csv.reader(csvfile), 1)[0]

    # Get yesterday's date and last logged date in both datetime format and in string format
    # Data from today is incomplete and not logged. So we are interested in yesterday's date.

    last_log_date = datetime.datetime.strptime(temp[0],"%Y-%m-%d")
    today = datetime.datetime.today()
    yesterday = today - datetime.timedelta(days=1) 

    # Get latest fitbit daily data
    df = get_latest_fitbit_data(authd2_client,last_log_date,yesterday)

    # Add latest daily data to daily data csv file
    write_fitbit_data_to_csv(df,'Data/Daily_Fitbit_Data.csv','Data/Last_log_date.csv')
    write_fitbit_data_to_csv(df,'Data/Data_master.csv','Data/Last_log_date.csv')
    
    return df
    

##########################################################################################
#Further Functions

def get_last_row(csv_filename):
    '''
    Example Usage:
        get_last_row('Data/Last_log_date.csv')[0]
        '''
    with open(csv_filename, 'r') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow
    


def log(func):
  '''decorator logging function'''
  def wrapper(*args, **kwargs):
    with open ('/Users/sablelevy/GitHub/fitbit_analyzer/Data/function_log.txt', 'a') as f:
      f.write('Called function ' + func.__name__ + ' with '+ ' '.join([str(arg) for arg in args]) + ' at ' + str(datetime.datetime.now()) + '\n')
    val = func(*args, **kwargs)
    return val
  return wrapper    
    
    
def save_plt(func):
  '''save a plot with a unique filename matching the function_ind'''
  ind = 0
  cache = {}
  def wrapper(*args):
    nonlocal ind 
    ind += 1
    plt = func(*args)
#    if func.__name__ not in cache:
    cache[func.__name__]=ind
        
    plt.savefig('Data/Images/{}_{}.png'.format(func.__name__,cache[func.__name__]), dpi=300)
    return plt
  return wrapper


@save_plt
def get_lmplot(df,x,y):
    '''Generates an lmplot given an x and y variable, 
    with subplot and hue based on 'year' 
    
    Args: 
        df (pandas dataframe): The data .
        x: x-axis variable
        y: y-axis variable
        
    Returns:
        lmplot 
        
    Example Usage:
        get_lmplot(df_latest,'steps','weight')
    '''    
    lm = sns.lmplot(data = df, x=x,
                          y=y, fit_reg=True, hue='year', 
                          scatter=True, col='year')
    return lm


@log
def apply_min_filter(df, filter_dic):
    '''Cleans a dataframe by applying min value filters to any number of columns
    
    Args: 
        df (pandas dataframe): The data to clean.
        filter_dic (dic): A dictionary with key-value pairs that specify the 
        name of the column and min value filter to impose.
        
    Returns:
        pandas dataframe: the filtered dataframe
        
    Example Usage:
        my_filter_dic = {'steps': 500, 'sleep_minutesAsleep': 500}
        df_filtered=(apply_min_filter(df, my_filter_dic))
    '''
    df_clean = df.loc[(df[list(filter_dic)] >= pd.Series(filter_dic)).all(axis=1)]
    return df_clean

@log
def apply_max_filter(df, filter_dic):
    '''Cleans a dataframe by applying max value filters to any number of columns
    
    Args: 
        df (pandas dataframe): The data to clean.
        filter_dic (dic): A dictionary with key-value pairs that specify the 
        name of the column and min value filter to impose.
        
    Returns:
        pandas dataframe: the filtered dataframe
        
    Example Usage:
        my_filter_dic = {'steps': 500, 'sleep_minutesAsleep': 500}
        df_filtered=(apply_max_filter(df, my_filter_dic))
    '''
    df_clean = df.loc[(df[list(filter_dic)] <= pd.Series(filter_dic)).all(axis=1)]    
    return df_clean

@log
def trim_dates(df, date_range=None):
    '''Trims a dataframe by a given date range and returns the trimmed dataframe 
    with a datetime index (whether or not the input df had a datetime index).
    
    Args: 
        df (pandas dataframe): The data to trim; must have 'Date' col if no DatetimeIndex.
        date_range (list): A list such as [start_date, end_date] 
            in the form['yyyy-mm-dd','yyyy-mm-dd']  
            If none given, returned df is untrimmed with datetime index
        
    Returns:
        pandas dataframe: the trimmed dataframe with a datetime index
        
    Example Usage:
        date_range = ['2018-05-25','2020-05-25']
        df_trimmed = trim_dates(df, date_range)
        df_with_dt_index = trim_dates(df)
    '''
    #Copying so changes aren't made to input df
    df=df.copy(deep=True)

    if type(df.index) == pd.core.indexes.datetimes.DatetimeIndex:
        if date_range == None:
            return df
        else:
            df_trimmed = df.loc[date_range[0] : date_range[1]]
    elif 'Date' in df.columns:
        if date_range == None:
            #setting date range to span the first and last entry in the Date column
            last_date=df['Date'].shape[0]-1
            date_range = [df['Date'][0], df['Date'][last_date]]
        df.loc[:,'Date'] = pd.to_datetime(df.loc[:,'Date'])    
        df_ind = df.set_index('Date').loc[date_range[0] : date_range[1]]
        df_trimmed = df_ind.reset_index()
    else:
        raise ValueError('df must contain `Date` column or have a DatetimeIndex.')
    
    return df_trimmed

@log
def trim_cols(df, col_dic):
    '''Trims a dataframe either by keeping only specified columns or by
    omitting them.
    
    Args:
        df (pandas dataframe): The dataframe to trim
        col_dic (dic): 
            Must be of form: {'cols_to_keep': ['col1','col2']} or
            {'cols_to_drop': ['col1']}
            
    Returns:
        pandas dataframe: the trimmed dataframe 
            
    Example Usage:
        col_dict = {'cols_to_drop': ['steps','dist']}
        new_df = trim_cols(df, col_dict)
    
    Notes:
        If col_dic has both 'cols_to_keep' and 'cols_to_drop' keys, 'cols_to_drop'
        is ignored.
    '''
    #Copying so changes aren't made to input df
    df=df.copy(deep=True)
    
    if 'cols_to_keep' in col_dic.keys():
        #listing only those columns that are in the input df
        cols_to_keep = [col for col in col_dic['cols_to_keep'] if col in df.columns]
        df = df[cols_to_keep]
    elif 'cols_to_drop' in col_dic.keys():
        #listing only those columns that are in the input df
        cols_to_drop = [col for col in col_dic['cols_to_drop'] if col in df.columns]
        df = df.drop(columns=cols_to_drop)    
    else:
        raise ValueError ('Dictionary parameter must contain one of two keys: \
                          cols_to_drop, cols_to_keep.')
    
    return df


@save_plt
def get_pairplot(df, x_vars=None, y_vars=None, custom_args = {'kind':'reg', 'palette': sns.color_palette("cubehelix", 5), 'hue': 'year', 'diag_kind': 'auto'}):
    '''Generate customizable pairplots for any number of xvars and yvars
    
    Args:
        df (pandas DataFrame): 
        custom_args (dictionary): specify customizable features
        x_vars (list): list of x variable(s) to plot
        y_vars (list): list of y variable(s) to plot
        
    Returns:
        Seaborn Pairplot
        
    Example Usage:
        get_pairplot(df, ['steps','weight','dist'], ['steps','weight','dist'], 
             {'kind':'scatter', 'palette': sns.color_palette("husl", 5),
             'hue': 'year', 'diag_kind': 'kde'})
        get_pairplot(df, ['steps','weight','dist'], ['steps','weight','dist'])
        get_pairplot(df)
        
    Notes:
        Copy default custom_args and replace only what you want to change.
        
    '''
    fig = sns.pairplot(data = df, 
                       x_vars=x_vars,
                       y_vars=y_vars,
                       kind=custom_args['kind'],
                       diag_kind=custom_args['diag_kind'],
                       hue=custom_args['hue'], 
                       palette = custom_args['palette'],
                       height=4)

    return fig


@save_plt
def get_corr_heatmap(df, cmap = 'gnuplot'):
    '''Display a correlation heatmap of specified columns from input df
    
    Args:
        df (pandas dataframe):
        columns (list): list of column names to include.  Uses all by default
        cmap (color palette): Possible values are: Accent, Accent_r,
         Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, 
         Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r,
         Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2,
         Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, 
         Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, 
         RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, 
         Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, 
         afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, 
         bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, 
         cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, 
         gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern,
         gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r,
         hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, 
         mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, 
         prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, 
         summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, 
         terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, 
         vlag_r, winter, winter_r
        
    Returns:
        Seaborn correlation heatmap of specified columns from input df
        
    Example Usage:
        get_corr_heatmap(df_latest, 'bwr')
        
    Notes:
        If we don't use get_figure() method, we get error from save_plt:
            AttributeError: 'AxesSubplot' object has no attribute 'savefig'
            
        
        
    '''    
    heatmap = sns.heatmap(df.corr(), cmap=cmap, cbar=True, linewidth=.2)
    plt.margins(5)
    return heatmap.get_figure()

    
###############################################################
CLIENT_ID = 'REDACTED'
CLIENT_SECRET = 'REDACTED'
redirect_uri = 'http://127.0.0.1:8080/'
fitbit_date_format = '%Y-%m-%d'
server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
server.browser_authorize()
ACCESS_TOKEN = str(server.fitbit.client.session.token['access_token'])
REFRESH_TOKEN = str(server.fitbit.client.session.token['refresh_token'])
#Here we store the token fetched above and initialize the auth2_client object with that token so that we can make authenticated calls.
auth2_client = fitbit.Fitbit(CLIENT_ID, CLIENT_SECRET, oauth2=True, access_token=ACCESS_TOKEN, refresh_token=REFRESH_TOKEN)
auth2_client.API_VERSION = 1.2 #this has to be manually done because the python library defaults to ver 1.
###############################################################


###Run this code to gather new daily data
obtain_write_new_data(auth2_client)
df_latest = pd.read_csv('Data/Data_master.csv', parse_dates=['Date']).sort_index(ascending=False)



class fitbit_user:
    
    def __init__(self, df, date_range=None):        
        #Make a copy of the df, to distinguish changes made within the class
        self.df = df.copy(deep=True)
        self.df['sleep_start'].fillna('00:00', inplace = True)
        self.df['datetime']=pd.to_datetime(self.df['Date'])
        self.col_list = list(self.df.columns)
        self.df['day'] = pd.to_datetime(self.df['Date']).dt.day_name()
        self.df['year'] = pd.to_datetime(self.df['Date']).dt.year
        self.df['month'] = pd.to_datetime(self.df['Date']).dt.month
        self.df['year'] = pd.to_datetime(self.df['Date']).dt.year
        self.df['sleep_start_str'] = self.df['Date'].astype(str) +' ' + self.df['sleep_start'].astype(str)
        self.df['sleep_start_dt'] = pd.to_datetime(self.df['sleep_start_str'], format='%Y-%m-%d %H:%M')
        
        self.df_date_index = df.set_index('Date')

        if date_range:
            self.df = self.df_date_index.loc[date_range[0] : date_range[1]]

    def get_day(self, date):
        return self.df[self.df['Date']==date]





    
    
