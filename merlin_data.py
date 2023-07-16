import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
plt.style.use('fivethirtyeight')
five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
]
sns.set_palette(five_thirty_eight)
from tqdm import tqdm
import re
from datetime import datetime 


def cleaning_df(df):
    
    """
        Returns cleansed dataframe
            Parameters:
                    df(dataframe)
            Returns:
                    df(dataframe)
    """
    
    df.rename(columns=lambda x:x.lower().strip().replace(' ','_'),inplace=True)
    df['date']=pd.to_datetime(df['date'])
    df.set_index('date',inplace=True)
    
    return df 

def check_missing(df):
    
    
    """
    Check null value and return total count and percentage of columns which contains null
    Parameters:
           df(dataframe)
    Returns:
           df(dataframe)


    """
    flag=df.isnull().sum().any()
    if flag==True:

        total=df.isnull().sum().sort_values(ascending=False)
        percent=(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
        missing=pd.concat([total,percent],axis=1,keys=['total','percent'])

        data_type=[]
        for col in df.columns:
            dtype=str(df[col].dtype)
            data_type.append(dtype)
        missing['Type']=data_type
        return missing
    else:
        return(False) 
def unstack_df_by_year(df):
    """
    Create '%m-%d' and 'year' columns as new index and pivot 'year' column, in unstack 'year' is in level -2 or level 0 
    Parameters:
           df(dataframe)
    Returns:
           df(dataframe)


    """
    tmp=df.copy()
    tmp=tmp.reset_index()
    tmp['year']=tmp.reset_index()['date'].dt.year
    tmp['date']=tmp.reset_index()['date'].dt.strftime('%m-%d')
    
    unstacked=tmp.set_index(['year','date'])['visitation'].unstack(-2)
    return unstacked

def unstack_df_by_year_month(df):
    """
    Create '%m' and 'year' columns as new index and pivot 'year' column , in unstack 'year' is in level -2 or level 0
    Parameters:
           df(dataframe)
    Returns:
           df(dataframe)


    """
    tmp=df.copy()
    tmp=tmp.reset_index()
    tmp['year']=tmp.reset_index()['date'].dt.year
    tmp['month']=tmp.reset_index()['date'].dt.month
    tmp=tmp.set_index('date').groupby(['year','month'])['visitation'].sum().unstack(-2)

    return tmp

def unstack_df_by_year_quarter(df):
    """
    Create 'quarter' and 'year' columns as new index and pivot 'year' column , in unstack 'year' is in level -2 or level 0
    Parameters:
           df(dataframe)
    Returns:
           df(dataframe)


    """
    tmp=df.copy()
    tmp=tmp.reset_index()
    tmp['year']=tmp.reset_index()['date'].dt.year
    tmp['quarter']=tmp.reset_index()['date'].dt.quarter
    tmp=tmp.set_index('date').groupby(['year','quarter'])['visitation'].sum().unstack(-2)

    return tmp

def create_new_features(df):
    """
    Create new features for forcasting and encode the cyclical feature
    Parameters:
           df(dataframe)
    Returns:
           df(dataframe)


    """
    
    df=df.reset_index()
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['day']=df['date'].dt.day
    df['day_of_year']=df['date'].dt.dayofyear
    df['week_of_year']=df['date'].dt.isocalendar().week
    df['quarter']=df['date'].dt.quarter
    df.set_index('date',inplace=True)
    month_in_year = 12
    df['month_sin'] = np.sin(2*np.pi*df['month']/month_in_year)
    df['month_cos'] = np.cos(2*np.pi*df['month']/month_in_year)
    week_in_year=52
    df['week_sin'] =np.sin(2*np.pi*df['week_of_year']/week_in_year)
    df['week_cos'] =np.cos(2*np.pi*df['week_of_year']/week_in_year)
    return df