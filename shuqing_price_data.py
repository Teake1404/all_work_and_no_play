import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
from io import StringIO
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
# plt.style.use('fivethirtyeight')
# five_thirty_eight = [
#     "#30a2da",
#     "#fc4f30",
#     "#e5ae38",
#     "#6d904f",
#     "#8b8b8b",
# ]
# sns.set_palette(five_thirty_eight)
# from tqdm import tqdm
import re
# import plotly.express as px
# import plotly.graph_objs as go
# import plotly.io as pio
# pio.renderers.default = 'iframe'
# from plotly.subplots import make_subplots
# from scipy.optimize import minimize
# from sklearn.preprocessing import StandardScaler
from bifrost.auth.drive import MSDrive
from collections import Counter
import math

class Preprcessing:
    """
    A class to clean and check null for dataframe
    
    Attributes
    ----------
    df: dataframe
        the df to be processed 
    
    Methods
    -------
    clean_df(df):
    Returns cleaned df 
    
    check_missing(df):
    Returns summary of missing values
    
    
    
    """
    
    def __init__(self,df):
        self.df= df
    
    def clean_df(self):
        """
        Returns cleansed dataframe
            Parameters:
                    df(dataframe)
            Returns:
                    df(dataframe)
 
        """
        self.df.rename(columns=lambda x: re.sub("\W+", "", x).lower(), inplace=True)

        date_columns=[c for c in self.df.columns if 'start_date' in c]

        date_columns.append(self.df.columns[2]) # this df.columns[2] is 'approved_at'

        for d in date_columns:
            self.df[d]=pd.to_datetime(self.df[d]).dt.strftime('%Y-%m-%d')
        
        self.df=self.df.replace('',None)
        
        return self.df 
    
    def check_missing(self):
        """
        Check null value and return total count and percentage of columns which contains null
        Parameters:
               df(dataframe)
        Returns:
               df(dataframe)


        """
        flag=self.df.isnull().sum().any()
        if flag==True:
            total=self.df.isnull().sum().sort_values(ascending=False)
            percent=((self.df.isnull().sum()/self.df.isnull().count())*100).sort_values(ascending=False)
            missing=pd.concat([total,percent],axis=1,keys=['total','percent%'])

            data_type=[]
            for col in self.df.columns:
                dtype=str(self.df[col].dtype)
                data_type.append(dtype)
            missing['Type']=data_type
            return missing
        else:
            return(False) 
        
    
    def contract_drop_missing_fillna(self):
        percent=((self.df.isnull().sum()/self.df.isnull().count())*100).sort_values(ascending=False)
        missing=pd.DataFrame(percent).rename(columns={0:'percent'})
        not_to_drop=['ship_to','product']
        
        self.df.drop([c for c in missing[missing['percent']>60].index if c not in not_to_drop],axis=1,inplace=True)
       
        self.df.drop(self.df[self.df.price_basis=='0.000000000'].index,axis=0,inplace=True)
        
        return self.df




    
def convert_split_cols(df,cols):
    """
    Convert numerical columns to float and split 'notification_index_fx_rate' into two parts, value and currency
        Parameters:
               df(dataframe, numerical_columns)
        Returns:
               df(dataframe)
    
    """
    for c in cols:
        df[c]=df[c].astype('float')
    
    df.drop(df[df.price_basis=='0.000000000'].index,axis=0,inplace=True)
    df['notification_index_fx_rate_value']=df.notification_index_fx_rate.str.split().str[0]
    df['notification_index_fx_rate_currency']=df.notification_index_fx_rate.str.split().str[1]
    
    df['start_date']=pd.to_datetime(df.start_date)    
    df['month_number']=df['start_date'].dt.month
    df['year']=df.start_date.dt.year
    df['month']=df.start_date.dt.strftime('%B')
    df['product_hierarchy_pa']=pd.Series([int(i) for i in df['product_hierarchy_pa'].fillna(0)])
    df['product']=pd.Series([int(i) for i in df['product'].fillna(0)])
    df['product']=df['product'].astype(str)
    df['end_date']=df['end_date'].astype(str).replace('9999-12-31 00:00:00','2050-12-31')
    df['end_date']=pd.to_datetime(df['end_date'],format='%Y-%m-%d')
    return df

def drop_missing_fillna(df):
    percent=((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending=False)
    missing=pd.DataFrame(percent).rename(columns={0:'percent'})
    not_to_drop=['sold_to',
     'sold_to_desc', 'product',
     'product_desc','index_value_raw',
       'index_value_raw_previous', 'fx_index_rate', 'index_uom',
       'index_currency', 'index_start', 'index_end']

    df.drop([c for c in missing[missing['percent']>60].index if c not in not_to_drop],axis=1,inplace=True)
    return df


def contract_convert_df(df):
    """
    Convert contract df to the right format
    
    """
    cols_to_drop=['price_decimal','index_value_raw', 'index_value_raw_previous', 'fixed_price_negotiated',
       'diff1_negotiated', 'diff2_negotiated', 'diff3_negotiated',
       'diff4_negotiated', 'discount_negotiated', 'fx_fixed_price_rate',
       'fx_index_rate', 'fx_diff1_rate', 'fx_diff2_rate', 'fx_diff3_rate',
       'fx_diff4_rate', 'fx_discount_rate', 'fx_type_index', 'fx_type_diff1',
       'fx_type_diff2', 'fx_type_diff3', 'fx_type_discount',
       'fixed_price_uom', 'index_uom', 'diff1_uom', 'diff2_uom', 'diff3_uom',
       'diff4_uom', 'discount_uom', 'fixed_price_currency', 'index_currency',
       'diff1_currency', 'diff2_currency', 'diff3_currency', 'diff4_currency',
       'discount_currency', 'contract_start', 'contract_end', 'index_start',
       'index_end', 'notification_index_basis',
       'notification_index_basis_previous', 'notification_index_basis_change',
       'notification_negotiated_diff1', 'notification_price_diff',
       'notification_index_fx_rate', 'notification_diff1_fx_rate',]
    df.drop(cols_to_drop,axis=1,inplace=True)
    
    float_cols=['price_basis','diff1','diff2','diff3','diff4','discount','net_price']
    for c in float_cols:
        df[c]=df[c].astype('float')
        
    df['start_date']=pd.to_datetime(df.start_date)    
    df['month_number']=df['start_date'].dt.month
    df['year']=df.start_date.dt.year
    df['month']=df.start_date.dt.strftime('%B')
    
    df['product_hierarchy_pa']=pd.Series([int(i) for i in df['product_hierarchy_pa'].fillna(0)])
    df.loc[df['product']=='NA','product']=0
    df['product']=pd.Series([int(i) for i in df['product'].fillna(0)])
    df['product']=df['product'].apply(lambda x:str(x).lstrip('0'))
    df['end_date']=df['end_date'].astype(str).replace('9999-12-31 00:00:00','2050-12-31')
    df['end_date']=pd.to_datetime(df['end_date'],format='%Y-%m-%d')   
    
    return df 
    
    
    

# def df_fillna(df):
#     """
#     fillna values with each group's median values, groups are defined by 'price_record_id','price_item_id','band_price'. For those still na values, fillna       with 0
#         Parameters:
#                df(dataframe)
#         Returns:
#                df(dataframe)
    
#     """

    
#     missing_fillna=df.isnull().sum().sort_values(ascending=False)
#     missing_fillna=pd.DataFrame(missing_fillna).rename(columns={0:'num'})
#     fillna_cols=missing_fillna.loc[missing_fillna.num>0].index
    
#     num_cols=df[fillna_cols].select_dtypes(include=np.number).columns
#     cat_cols=[c for c in fillna_cols.tolist() if c not in num_cols]

#     # Fillna with each group's median values, groups are defined by 'price_record_id','price_item_id','band_price'. For those still na values, fillna with 0
#     df[num_cols]=df[num_cols].fillna(df.groupby(['price_record_id','price_item_id','band_price']).transform('median'))
#     df[num_cols]=df[num_cols].fillna(0)
    
#     for c in cat_cols:
#         df[c]=df[c].fillna(df[c].mode()[0])
#     return df 

def check_state(df):
    """
    Check if there is still na in df before visualisation 
    Parameters:
               df 
    Returns:
            print out null list
            False (if there is no null)
    
    """
    flag=df.isnull().sum().any()
    if flag==True:
        print(df.isnull().sum())
    else:
        return False

def create_buildup_median_graph(df,x,y,title):
    """
    Create standardised graphs to visualise basic trends between x and y
    
    
    """
    fig=go.Figure(data=go.Scatter(x=df[x],
                              y=df[y],
                              marker_color='blue'      
                             ))
    fig.update_layout({'title':f'{title}',
                      'xaxis':{'title':x},
                      'yaxis':{'title':y},
                      'showlegend':False
                     
                      })
    return fig.show('iframe')
    

# def create_median_bar_by_variable(df,x,y,color,title,cat,yaxis):
    
#     fig=px.bar(df,x=x,y=y, color=color)
#     fig.update_xaxes(categoryorder='array', categoryarray= cat)
#     fig.update_layout({"title": f'{title}',
#                   "yaxis":{'title':f'{yaxis}'}}
#                        )
#     fig.add_annotation(dict(font=dict(color='yellow',size=25),text='WORKING IN PROGESS',y=100000))

#     return fig.show('iframe')

def create_stacked_bar(df,x,y,color,title,cat):
    fig=px.bar(df,x,y,color)
    fig.update_xaxes(categoryorder='array', categoryarray= cat)
    fig.update_layout({'title':f'{title}'})
    return fig.show('iframe')

def plot_box_seasonality(df,x_year,y,x_month):
    fig,ax=plt.subplots(1,2,figsize=(15,6))
    sns.boxplot(data=df,x=x_year,y=y,ax=ax[0], palette="turbo")
    currency_value=df['currency_code'].mode()[0]
    qty_type= df['qty_mapping'].mode()[0]
    ax[0].set_title(f'Year-wise Box Plot\n(The Trend) in {currency_value} measured in {qty_type}',fontsize=15,loc='center',fontdict=dict(weight='bold'))
    ax[0].set_xlabel(x_year, fontsize = 15, fontdict=dict(weight='bold'))
    ax[0].set_ylabel(y, fontsize = 15, fontdict=dict(weight='bold'))
    
    sns.boxplot(data=df,x=x_month,y=y, ax=ax[1], palette="turbo")
    ax[1].set_title(f'Month-wise Box Plot\n(The Seasonality) in {currency_value} measured in {qty_type}', fontsize =15, loc='center', fontdict=dict(weight='bold'))
    ax[1].set_xlabel(x_month, fontsize =16, fontdict=dict(weight='bold'))
    ax[1].set_ylabel(y, fontsize = 16, fontdict=dict(weight='bold'))
    ax[1].tick_params(axis='x', labelrotation=45)
    
    return plt.show()



class Mapping:
    """
    This class is used to map BAND_PRICE_ID, PRODUCT_HIER_ID in invoice data from insighthub using the logic in PROS data
    There are three dfs involved, df(invoice_df), band_df(which contains band_price_id) and hier_df (which contains product_hier_id)      
    """
    
    def __init__(self,df,band_df,hier_df):
        self.df=df
        self.band_df=band_df
        self.hier_df=hier_df
    
    def mapping_price_band(self):
        
        """
        This is the BAND_PRICE_ID mapping function for invoice data

        Using [price_group] [price_list_type] in band_price_mapping table to map and ingest [band_price_id] in df


        """
        self.df['price_list_type']=self.df['price_list_type'].apply(lambda x: str(x).strip(' '))
        self.band_df.rename(columns=lambda x:x.lower().replace('_id',''),inplace=True)
        _res=[]
        for r in self.band_df.to_dict(orient='records'):
            _res+=[{(r['price_group'],r['price_list_type']):r['band_price']}] ## create the df that contains a column that has key of 
                                                                              ## key[price_group,price_list_type] values of band_price 
        dict1={}

        for k, v in [(k,v) for x in _res for (k,v) in x.items()]:
            dict1[k]=v                                            # create a dict that consists of key of [price_group,price_list_type] and value of                                                                            # band_price

        tmp=self.df[['price_group','price_list_type']].copy()

        tmp['band_price_id']=tmp.apply(tuple,axis=1).map(dict1)

        band_price_list=tmp.apply(tuple,axis=1).map(dict1)

        self.df['band_price_id']=band_price_list
        return self.df

    def clean_product_hier(self):
        
        self.hier_df.rename(columns=lambda x: x.lower(),inplace=True)
        self.hier_df['product_id']=self.hier_df['product_id'].fillna(0)
        int_values=pd.Series([int(i) for i in self.hier_df['product_id']]) # because the value is like 100002910.0 with trailing decimals
        self.hier_df['product_id']=int_values
        return self.hier_df 

    def create_product_hier(self):
        """
        create the mapping to map product_hier_id with product_id

        """
        self.df=self.mapping_price_band()
        self.hier_df=self.clean_product_hier()
        for c in self.hier_df.columns[1:].tolist():# they are [product_hier_id,product_id]
            self.hier_df[c]=self.hier_df[c].astype(str)

        product_mapping=dict(zip(self.hier_df['product_id'],self.hier_df['product_hier_id']))
        self.df['trx_pro_key_new']=self.df['trx_pro_key'].apply(lambda x:x.lstrip('0'))
        self.df['product_hier_id']=self.df['trx_pro_key_new'].map(product_mapping)
        
        
        # drop columns and return clean df 
        columns_to_drop=['source_system','billing_date','billing_document','billing_document_prefix','billing_item','flight_number',
         'aircraft_registration'  ,'trx_billto_key','trx_payer_key','trx_cust_key', 'trx_pro_key','order_creation_date' ,
                 'gcr_ii_exc_ind','gcr_ii_exc_reason','gcr_dc_exc_ind','gcr_dc_exc_reason','cust_key','card_number','mdm_partner_id','soldto_mdm_partner_id']
        self.df.drop(columns_to_drop,axis=1,inplace=True)
        
        # prepare self.df to be ready for join later on 
        self.df['trx_loc_key']=self.df['trx_loc_key'].str.replace('PRE','')
        self.df['band_price_id']=self.df['band_price_id'].str.lstrip('0')
        self.df['band_price_id']=self.df['band_price_id'].str.replace('TBD','').replace('',0).fillna(0).astype(int)  
        self.df['product_hier_id']=self.df['product_hier_id'].fillna(0).astype(int)

        return self.df

def pros_fill0_product_product_hierarchy_pa(df):
    """
    Use product_code to fill in 0 for product and product_hier_pa
    """
    tmp=df[['product_code','product','product_hierarchy_pa']]
    mapping=tmp.drop_duplicates().set_index('product_code')['product'].to_dict()
    tmp.loc[tmp['product']==0,'product']=tmp.loc[tmp['product']==0,'product_code'].map(mapping)
    
    hier_mapping=tmp.drop_duplicates().set_index('product_code')['product_hierarchy_pa'].to_dict()
    tmp.loc[tmp['product_hierarchy_pa']==0,'product_hierarchy_pa']=tmp.loc[tmp['product_hierarchy_pa']==0,'product_code'].map(hier_mapping)
    
    return tmp 


def create_sub_pros_df_for_joins(df, length_limit):
    """
    Splits a DataFrame into sub-DataFrames that do not exceed the specified length limit.
    
    Parameters:
    df : pd.DataFrame
        The input DataFrame to be split.
    length_limit : int
        The maximum allowed length of any sub-DataFrame.
    
    Returns:
    List of pd.DataFrame
        A list of sub-DataFrames, each of which does not exceed the specified length limit.
    """
    total_rows = df.shape[0]
    num_chunks = math.ceil(total_rows / length_limit)
    
    sub_dfs_dict={}
    
    for i in range(num_chunks):
        start_idx = i * length_limit
        end_idx = min((i + 1) * length_limit, total_rows)
        sub_df = df.iloc[start_idx:end_idx, :]
         
        # Create a dynamic name for each sub-DataFrame
        sub_df_name= f"sub_df_{i+1}"
        sub_dfs_dict[sub_df_name]=sub_df
    
    return sub_dfs_dict

def merge_process_pros_df(df):
    """
    Clean pro_df to be processed for merge with invoice df later 
    """
    
    df["product_hierarchy_pa"] = pd.Series(
    [int(i) for i in df["product_hierarchy_pa"].fillna(0)]
    )
    df["product"] = pd.Series([int(i) for i in df["product"].fillna(0)])
    df["product"] = df["product"].astype(str)
    df["end_date"].replace("9999-12-31 00:00:00", "2030-12-31", inplace=True)
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y-%m-%d")
    return df 

def merge_pros_loc_seg_remove_duplicated_loctionid(pros,df):
    """
    Remove dudplicated locationid with the same mdm_dp_identifier 
    merge pros with loc_seg
    
    """
    count_loc=df.groupby(['mdm_dp_identifier'],as_index=False)['soluslocation'].count()

    if count_loc.loc[count_loc.soluslocation>1]['soluslocation'].count()>0:

        cols_to_keep=df.columns.tolist()

        cols_to_keep=[c for c in cols_to_keep if c not in ['deliverypointid','operationtype','soluslocation','deliverypointstatus']]

        tmp=df.drop_duplicates(subset=cols_to_keep,keep='last',inplace=False)

        df_tmp=pros.merge(tmp,left_on='delivery_point',right_on='mdm_dp_identifier',how='left')
    return df_tmp


def check_band_price_multiple_delivery_points_per_shipping_points(df):
    """
    remove duplicates for the same price with variation attributes
    """
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0',axis=1,inplace=True)
    
    df=df.sort_values(by=['approved_at'],ascending=False)
    count_delivery_point=df.groupby(['start_date','end_date','shipping_point','product',
                                           'product_hierarchy_pa','net_price','band_price','band_price_desc'],as_index=False)['delivery_point'].count()
    count_more_than_one=count_delivery_point.loc[count_delivery_point.delivery_point>1,'band_price'].count()
    
    if count_more_than_one!=0:
        
        
        columns_keep=df.columns
        columns_keep=[ c for c in columns_keep if c not in ['price_record_id', 'price_item_id','approved_at','delivery_point','delivery_point_desc',
                                                   'product_code','product_code_desc','uom_code','index_uom','index_currency',
                                                            'index_code_ref','index_code_ref_desc','index_code','index_code_desc',
                                                            'index_value_raw','index_value_raw_previous','fx_index_rate','location','location_desc']]
        tmp=df.drop_duplicates(subset=columns_keep,keep='first',inplace=False)
        return tmp 

    else:
        return df
    


def merge_pros_invoice(df,pros_df):
    """
    Join PROS df with invoice df (named as df) using two levels of joins, first level using [product] from PROS and [trx_pro_key_new] from invoice
    Second join using [product_hierarchy_pa] from PROS and [product_hier_id] from invoice 
      
    First level join result should return df with not null erp_id, second level join happens where no match is found on product_id level so product in PROS     data should be null (0 if it is imputed)
    
    After getting two levels of joins, we concatenate them together to return final merge_df 
    
    For both joins delivery_date in invoice df should be between the [start_date,end_date] of pros df 
    
    """
    first_left_on=['band_price_id','delivery_point_code','trx_pro_key_new','trx_loc_key','sales_organisation']
    first_right_on=['band_price','shipping_point','product','supply_loc','sales_org']

    left_on=['band_price_id','delivery_point_code','product_hier_id','trx_loc_key','sales_organisation']
    right_on=['band_price','shipping_point','product_hierarchy_pa','supply_loc','sales_org']

    first_merge_df=df.reset_index().merge(pros_df,how='left',left_on=first_left_on,right_on=first_right_on).set_index('index')
    no_erp_df=first_merge_df.loc[first_merge_df['erp_id'].isnull()]
    first_merge_df=first_merge_df.loc[first_merge_df['erp_id'].notnull()]
    first_merge_df=first_merge_df.loc[(first_merge_df['delivery_date']>=first_merge_df['start_date'])
                                          & (first_merge_df['delivery_date']<=first_merge_df['end_date'])]

    first_merge_list=first_merge_df.index.tolist()
    second_df=df.loc[~df.index.isin(first_merge_list)]

    second_merge_df=second_df.reset_index().merge(pros_df,how='left',left_on=left_on,right_on=right_on).set_index('index')
    second_merge_df=second_merge_df.loc[(second_merge_df['delivery_date']>=second_merge_df['start_date'])
                                      & (second_merge_df['delivery_date']<=second_merge_df['end_date'])]


    merge_df=pd.concat([first_merge_df,second_merge_df])


    merge_index_list=merge_df.index.tolist()

    no_match_df=df.loc[~df.index.isin(merge_index_list)]
    
    
    
    
    return merge_df

def remove_band_price_merge_duplicates(df,index_list):
    """
    remove duplicates for the same price with different UOM measurement
    """
    df=df.sort_values(by=['approved_at'],ascending=False) 
    index_df=df.loc[df.index.isin(index_list)]
    no_index_df=df.loc[~df.index.isin(index_list)]
        
    columns_keep=index_df.columns
    columns_keep=[ c for c in columns_keep if c not in ['price_record_id', 'price_item_id','approved_at',
                                                        'net_price','price_basis','uom_code','index_uom','index_currency','product_code','product_code_desc','product_hierarchy_pa_desc']]
    tmp=index_df.drop_duplicates(subset=columns_keep,keep='first',inplace=False)
    
    new=pd.concat([no_index_df,tmp])
    return new 
    


class merge_preprocessing:
    
    
    def __init__(self, df):
        self.df = df

    def clean_df(self):
        columns_to_drop = ["approved_at",
                          'trx_loc_key',
                         'last_modified_date',
                         'sum_key',
                         'erp_id',
                         'price_group',
                         'price_list_type',
                         'price_list_desc',
                         'sem_key',
                         'dim_loc_key',
                         'dim_loc_key',
                          'index_code',
                         'index_code_desc',
                         'index_code_ref',
                         'index_code_ref_desc',
                         'index_value_raw',
                         'index_value_raw_previous',
                         'fx_index_rate',
                         'index_uom',
                         'index_currency',
                         'index_start',
                         'index_end',
                        'trx_pro_key_new',
                         'product_hier_id',
                          'delivery_number',
                           'l1_core_marketing_rcop_lc','l1_core_marketing_rcop_usd','gross_profit_lc','gross_profit_usd','gross_profit_cso_lc','gross_profit_cso_usd',
                           'l4_cs_and_o_lc','l4_cs_and_o_usd','net_value_lc','net_value_usd'
                        ]
        self.df.drop(columns_to_drop, axis=1, inplace=True)

        return self.df

    def check_missing(self):
        self.df = self.clean_df()
        flag = self.df.isnull().sum().any()
        if flag == True:
            total = self.df.isnull().sum().sort_values(ascending=False)
            percent = (
                (self.df.isnull().sum()) / (self.df.isnull().count()) * 100
            ).sort_values(ascending=False)
            missing = pd.concat([total, percent], axis=1, keys=["total", "percent"])

            data_type = []
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                data_type.append(dtype)
            missing["Type"] = data_type
            return missing
        else:
            return False

    def drop_missing_fillna(self):
        percent = (
            (self.df.isnull().sum() / self.df.isnull().count()) * 100
        ).sort_values(ascending=False)
        missing = pd.DataFrame(percent).rename(columns={0: "percent"})
        not_to_drop = ["product_desc"]

        self.df.drop(
            [c for c in missing[missing["percent"] > 60].index if c not in not_to_drop],
            axis=1,
            inplace=True,
        )

        fillna_cols = ["product_hierarchy_pa_desc", "product_desc","customer_external_lens","region"]

        for c in fillna_cols:
            self.df[c].fillna("Unknown", inplace=True)
        self.df.fillna("Unknown", inplace=True)
        return self.df
    
    
def convert_merge_df(df):
    df["band_price_cat"] = df["band_price_desc"].str.split("-").str[0].str.rstrip()
    df.loc[
        df["band_price_cat"] == "STERLING CARD INTERNATIONAL AG", "band_price_cat"
    ] = "STERLING CARD INTERNATIONAL"
    
    num_cols=['price_basis','diff1','net_price','qty_in_m3',
     'qty_in_usg',
     'qty_in_litres',
     'net_value',
     'l2_gross_profit_lc',
     'l2_gross_profit_usd',
     'l3_revenue_lc',
     'l3_revenue_usd',
     'l3_cost_lc',
     'l3_cost_usd',
     'l4_customer_price_lc',
     'l4_customer_price_usd',
     'l4_invoice_line_item_lc',
     'l4_invoice_line_item_usd',
     'l4_other_costs_lc',
     'l4_other_costs_usd',
     'l4_pre_airfield_sh_lc',
     'l4_pre_airfield_sh_usd',
    'l4_taxes_excise_duty_lc',
    'l4_taxes_excise_duty_usd',
     'l4_pre_airfield_transport_lc',
     'l4_pre_airfield_transport_usd',
     'l4_on_airfield_costs_lc',
     'l4_on_airfield_costs_usd',
     'l4_purchase_price_lc',
     'l4_purchase_price_usd',
             ]

    for c in num_cols:
        df[c]=round(df[c],2)
        
    
    df['delivery_date']=pd.to_datetime(df['delivery_date'],format='%Y-%m-%d')

    df['year_month']=df['delivery_date'].apply(lambda x: x.strftime('%Y-%m'))
    
    df['start_date']=pd.to_datetime(df['start_date'],format='%Y-%m-%d')
    
    df['location_country_name']=df['location_desc']+'-'+df['country_name']
    
    
    return df

def convert_band_price_date_to_str(df):
    df['pricing_date']=df['pricing_date'].dt.date.astype('str')
    df['pricing_date']=df['pricing_date'].astype('str')
    df['delivery_date']=df['delivery_date'].dt.date.astype('str')
    df['delivery_date']=df['delivery_date'].astype('str')
    
    df['start_date']= df['start_date'].dt.date.astype('str')
    df['start_date']= df['start_date'].astype('str')
    
    df['end_date']=df['end_date'].dt.date.astype('str')
    df['end_date']=df['end_date'].astype('str')
    return df 
 
def qty_mapping_function(df):
    # creating qty mapping column which maps with right type of qty using uom_code column
    df['qty_in_hl']=(df.qty_in_litres/100)
    qty_mapping_dict={'UGL':'qty_in_usg',
            'M3':'qty_in_m3',
            'L':'qty_in_litres',
            'HL' :'qty_in_hl'        }
    df['qty_mapping']=df.uom_code.map(qty_mapping_dict)
    
    
    return df 

def generate_revenue_by_qty(df):
    """
    Create column revenue by multiplcation of 'net_price' and respective qty type
    
    
    """
    qty_columns=[ c for c in df.columns if c.startswith('qty')][:4]
    for c in qty_columns:
        name=f'revenue_{c}'
        df[name]=round(df[c]*df['net_price'],2)
        
    rev_mapping={'qty_in_usg':'revenue_qty_in_usg',
            
            'qty_in_hl':'revenue_qty_in_hl',
           'qty_in_m3' :'revenue_qty_in_m3',
           'qty_in_litres' :'revenue_qty_in_litres'}
    
    df['revenue_mapping']=df['qty_mapping'].map(rev_mapping)
    
    df.loc[df['currency_code']=='EUR','currency_code']=df.loc[df['currency_code']=='EUR','currency_code'].replace('EUR','EUR5')
    return df 


def remove_incremental_new_data_merge_df(df1,df2):
    """
    In order to save processing time for loading the whole dataset from scratch again, we only load incrementally 
    this function used to remove any duplicates 
    """
    pre_sale_doc=df1.sales_document.tolist()
    
    curr_sales_doc=df2.sales_document.tolist()
    new_curr_df=df2.loc[~df2.sales_document.isin(pre_sale_doc)]
    
    return new_curr_df
    
    


###### Publised band price data functions 

def clean_published_band_price_merge_df(df):
    
    columns_to_keep=df.columns.tolist()[:50]
    
    columns_to_keep=columns_to_keep+['qty_in_m3',
     'qty_in_usg',
     'qty_in_litres','qty_in_hl','qty_mapping','band_price_cat']


    df['has_transaction']=np.where(df['delivery_date'].notnull(),1,0)

    new_columns=columns_to_keep+['has_transaction']

    new_merge_df=df[new_columns]
    
    new_merge_df=new_merge_df.sort_values(by='net_price')

    new_merge_df=new_merge_df.sort_values(by='has_transaction')

    new_merge_df=new_merge_df.loc[new_merge_df.net_price!=0.00000]
    new_merge_df.drop_duplicates(inplace=True)
    columns_to_consider=new_merge_df.columns[:-1]
    
    tmp=new_merge_df.drop_duplicates(subset=columns_to_consider,keep='last')
    return tmp

def merge_published_band_price_trx(new_pros_df,df):
    first_right_on=['band_price_id','delivery_point_code','trx_pro_key_new','trx_loc_key','sales_organisation']
    first_left_on=['band_price','shipping_point','product','supply_loc','sales_org']

    right_on=['band_price_id','delivery_point_code','product_hier_id','trx_loc_key','sales_organisation']
    left_on=['band_price','shipping_point','product_hierarchy_pa','supply_loc','sales_org']
    
    first_merge_df=new_pros_df.reset_index().merge(df,how='left',left_on=first_left_on,right_on=first_right_on).set_index('index')
    second_merge_df=new_pros_df.reset_index().merge(df,how='left',left_on=left_on,right_on=right_on).set_index('index')
    merge_df=pd.concat([first_merge_df,second_merge_df])
    return merge_df

def convert_published_band_price_merge_df(df):
    df["band_price_cat"] = df["band_price_desc"].str.split("-").str[0].str.rstrip()
    df.loc[
        df["band_price_cat"] == "STERLING CARD INTERNATIONAL AG", "band_price_cat"
    ] = "STERLING CARD INTERNATIONAL"
    
    num_cols=['price_basis','diff1','net_price','qty_in_m3',
     'qty_in_usg',
     'qty_in_litres',
    
             ]

    for c in num_cols:
        df[c]=df[c].fillna(0)
        df[c]=round(df[c],2)
        
    
    df['approved_at']=pd.to_datetime(df['delivery_date'],format='%Y-%m-%d')

    
    
    return df


#####   Contract data functions 

def check_contract_multiple_delivery_points_per_shipping_points(df):
    """
    Another import function in here is to keep 1 record of price for each shipping_point and drop the rest different delivery_points 
    for per shipping_point, because for some shipping_points it will have multiple delivery_point and this will create duplicates later in the merge 
    """
    df=df.sort_values(by=['approved_at'],ascending=False)
    count_delivery_point=df.groupby(['tender','customer_grn','start_date','end_date','shipping_point','product','product_code',
                                           'product_hierarchy_pa','net_price','price_group_id','price_list_type_id'],as_index=False)['delivery_point'].count()
    count_more_than_one=count_delivery_point.loc[count_delivery_point.delivery_point>1,'customer_grn'].count()
    
    if count_more_than_one!=0:
        
        
        columns_keep=df.columns
        columns_keep=[ c for c in columns_keep if c not in ['price_record_id', 'price_item_id','delivery_point','delivery_point_desc','approved_at',
                                                           'index_code','index_code_desc','tender','tender_desc',
                                                            'sold_to_desc','customer_grn_desc','location_desc','shipping_point_desc','location','location_desc','index_class_desc','period']]
        tmp=df.drop_duplicates(subset=columns_keep,keep='first',inplace=False)
        return tmp 

    else:
        return df
    
def check_contract_new_after_removing_duplicates(df):
    count_delivery_point=df.groupby(['tender','customer_grn','start_date','end_date','shipping_point',
                                               'product','product_code','product_hierarchy_pa','net_price','price_group_id','price_list_type_id'],as_index=False)['delivery_point'].count()
    count_more_than_one=count_delivery_point.loc[count_delivery_point.delivery_point>1,'customer_grn'].count()
    if count_more_than_one!=0:
        return True
    else:
        return False
    
def check_merge_df_if_any_duplicates(merge_df):
    """
    check in merge_df if there is further potential duplicates
    to check the exact records uses merge_df.loc[merge_df.index==7269]
    
    
    """
    merge_index_list=merge_df.index.tolist()


    # sales_doc_list=merge_index_list+no_match_df_index_list

    # df_sales_list=df.index.tolist()

    counter_list=Counter(merge_index_list)

    count_2_number=Counter([i for i in counter_list.values()])[2]
    if count_2_number!=0:
        return count_2_number,[i for i,v  in counter_list.items() if v>=2]
    else:
        return False
    

def remove_contract_merge_df_duplicates(df,index_list):
    """
    remove the duplicated index from contract merge df
    
    """
    merge_df=df.loc[~df.index.isin(index_list)]
    
    return merge_df


def export_sharepoint(df,new_drive_id,new_item_path):
    """
    Export the files to Sharepoint location
    
    """
    csv_as_string = df.to_csv(index=True, encoding='utf-8') #
    file_object = csv_as_string.strip()
    file_object_bytes = file_object.encode("utf-8")
    drive = MSDrive()
    # Call MSDrive to save memory file to new location
    try:
        upload_status = drive.upload_memory_file_object(
            drive_id=new_drive_id,
            item_path=new_item_path,
            file_object=file_object_bytes,
        )            
        if upload_status:
            print("File Uploaded successfully")
        else:
            print("File upload failed")
    except Exception as e:
        print(f"Error: {str(e)}")
        
def download_from_sharepoint(new_drive_id,new_item_path):
    """
    download the csv from sharepoint 
    
    """
    drive = MSDrive()
    try:
        
    
        file_object = drive.data_content(
        drive_id=new_drive_id, item_path=new_item_path
        )
        if new_item_path.endswith("csv"):
            
            # Display CSV Content
            csv_data = StringIO(file_object.decode())

            df = pd.read_csv(csv_data)
            if 'Unnamed: 0' in df.columns:
                df.drop(['Unnamed: 0'],axis=1,inplace=True)

    except Exception as e:
        print(f"Error: {str(e)}")
        
    return df 

def preprocess_contract_invoice(df,contract):
    """fillna for missing iata using icao code so that this could be matched to PROS location"""
    # df.drop(df.loc[df['gcr_ii_exc_reason']=='Non-Fuel'].index,inplace=True)
    contract['product']=contract['product'].fillna(0).apply(lambda x:int(x)).astype(str)
    contract['customer_grn']=contract['customer_grn'].astype(str)
    return df,contract

def contract_merge_invoice(df, contract):
    first_left_on=['grn','sales_organisation','trx_loc_key','delivery_point_code','trx_pro_key_new']
    first_right_on=['customer_grn','sales_org','supply_loc','shipping_point','product']
    left_on=['grn','sales_organisation','trx_loc_key','delivery_point_code','product_hier_id']
    right_on=['customer_grn','sales_org','supply_loc','shipping_point','product_hierarchy_pa']
    
    first_merge_df=df.reset_index().merge(contract,how='left',left_on=first_left_on,right_on=first_right_on).set_index('index')
    first_merge_df=first_merge_df.loc[first_merge_df.price_record_id.notnull()]
    first_merge_df=first_merge_df.loc[(first_merge_df['delivery_date']>=first_merge_df['start_date'])
                                              & (first_merge_df['delivery_date']<=first_merge_df['end_date'])]
    first_merge_list=first_merge_df.index.tolist()
    
    
    second_df=df.loc[~df.index.isin(first_merge_list)]
    second_merge_df=second_df.reset_index().merge(contract,how='left',left_on=left_on,right_on=right_on).set_index('index')
    second_merge_df=second_merge_df.loc[(second_merge_df['delivery_date']>=second_merge_df['start_date'])
                                      & (second_merge_df['delivery_date']<=second_merge_df['end_date'])]
    merge_df=pd.concat([first_merge_df,second_merge_df])
    
    merge_index_list=merge_df.index.tolist()

    no_match_df=df.loc[~df.index.isin(merge_index_list)]
    
    merge_df=merge_df.drop_duplicates()
    return no_match_df, merge_df

def clean_contract_merge_df(df):
    cols_to_convert=['product_hierarchy_pa','product']

    cols_to_drop=['trx_pro_key_new','dim_loc_key']

    for c in cols_to_convert:
        df[c]=df[c].fillna(0).apply(lambda x:int(x))

    df=df.drop(cols_to_drop,axis=1)
    return df 

class contract_merge_preprocessing:
    
    
    def __init__(self, df):
        self.df = df

    def clean_df(self):
        columns_to_drop = [
            "approved_at",
            "pricing_date",
            "sales_organisation",
            "trx_loc_key",
            "last_modified_date",
            "sum_key",
            "delivery_method",
            "delivery_number",
            "fueling_method",
            "delivery_point_code",
            "erp_id",
            "band_price_id",
            "product_hier_id",         
            "price_group_id",
            "price_list_type_id",
            "price_group",
            "price_list_type",
            'l1_core_marketing_rcop_lc','l1_core_marketing_rcop_usd','net_value_lc',
            'net_value_usd',
            'gross_profit_lc','gross_profit_usd','gross_profit_cso_lc','gross_profit_cso_usd'
            
        ]
        self.df.drop(columns_to_drop, axis=1, inplace=True)

        return self.df

    def check_missing(self):
        self.df = self.clean_df()
        flag = self.df.isnull().sum().any()
        if flag == True:
            total = self.df.isnull().sum().sort_values(ascending=False)
            percent = (
                (self.df.isnull().sum()) / (self.df.isnull().count()) * 100
            ).sort_values(ascending=False)
            missing = pd.concat([total, percent], axis=1, keys=["total", "percent"])

            data_type = []
            for col in self.df.columns:
                dtype = str(self.df[col].dtype)
                data_type.append(dtype)
            missing["Type"] = data_type
            return missing
        else:
            return False

    def drop_missing_fillna(self):
        percent = (
            (self.df.isnull().sum() / self.df.isnull().count()) * 100
        ).sort_values(ascending=False)
        missing = pd.DataFrame(percent).rename(columns={0: "percent"})
        not_to_drop = ["product_desc"]

        self.df.drop(
            [c for c in missing[missing["percent"] > 60].index if c not in not_to_drop],
            axis=1,
            inplace=True,
        )

        self.df.fillna("Unknown", inplace=True)
        return self.df
    
def contract_convert_merge_df(df):
    
    num_cols=['price_basis','diff1','net_price','qty_in_m3',
     'qty_in_usg',
     'qty_in_litres',
     'net_value',
 'l2_gross_profit_lc',
       'l2_gross_profit_usd', 'l2_gross_profit_cso_lc','l2_gross_profit_cso_usd', 'l3_revenue_lc', 'l3_revenue_usd', 'l3_cost_lc',
       'l3_cost_usd', 'l4_customer_price_lc',
       'l4_customer_price_usd', 'l4_invoice_line_item_lc',
       'l4_invoice_line_item_usd', 'l4_taxes_excise_duty_lc',
       'l4_taxes_excise_duty_usd', 'l4_other_costs_lc', 'l4_other_costs_usd',
       'l4_pre_airfield_sh_lc', 'l4_pre_airfield_sh_usd',
       'l4_pre_airfield_transport_lc', 'l4_pre_airfield_transport_usd',
       'l4_on_airfield_costs_lc', 'l4_on_airfield_costs_usd',
       'l4_purchase_price_lc', 'l4_purchase_price_usd', 'l4_taxes_excise_duty_lc',
        'l4_taxes_excise_duty_usd',
              'price_basis', 'diff1',
       'diff2', 'diff3', 'diff4', 'discount', 'net_price']

    for c in num_cols:
        df[c]=round(df[c],2)
        
    
    df['delivery_date']=pd.to_datetime(df['delivery_date'],format='%Y-%m-%d')

    df['year_month']=df['delivery_date'].apply(lambda x: x.strftime('%Y-%m'))
    
    df['start_date']=pd.to_datetime(df['start_date'],format='%Y-%m-%d')
    
    
    
    
    return df

def contract_index_code_convert_pros(sample_index_code):
    """
    This function converts contract index codes to their corresponding 'pros' descriptions based on predefined rules.
    
    Parameters:
    - sample_index_code (list of str): A list of index codes to be converted.
    
    Returns:
    - list of str: A list of converted index codes in the pros language.
    
    Conversion Logic:
    - The function first checks the start of each code and maps it to a description.
    - Then, it checks the end of each code for additional mappings.
    - Finally, it combines the front and end descriptions.
    """
    
    # Mapping for the start of the code
    front_mapping = {
        'AL': 'Ad-Hoc Low', 
        'AH': 'Ad-Hoc High', 
        'X': 'PAP', 
        'S': 'Sterling Card International', 
        'L': 'Sterling Card Local', 
        'T': 'Premier International', 
        'RG': 'Reseller GA', 
        'RC': 'Reseller CA', 
        'P0': 'Platts Monthly', 
        'P1': 'Platts HalfMonthly', 
        'P2': 'Platts Weekly', 
        'DE': 'Germany Index', 
        'GR': 'Greece Index', 
        'CY': 'Cyprus Index', 
        'GB': 'UK Index'
    }
    
    # Mapping for the end of the code
    end_mapping = {
        'J': 'International Jet', 
        'AE': 'Local Avgas ', 
        'A': 'International Avgas', 
        'E': 'Local Jet', 
        'UAE': 'Unleaded Avgas ', 
        'F': 'F-34 (Jets)', 
        'FSI': 'Jet(FSI)'
    }
    
    # Process each code in the list
    converted_codes = []
    for code in sample_index_code:
        # Default values
        front_description = code
        end_description = ""
        
        # Check front part of the code
        for prefix, desc in front_mapping.items():
            if code.startswith(prefix):
                front_description = desc
                break
        
        # Check end part of the code
        for suffix, desc in end_mapping.items():
            if code.endswith(suffix):
                end_description = desc
                break
        
        # Combine front and end descriptions
        combined_description = f"{front_description} {end_description}".strip()
        converted_codes.append(combined_description)
    
    return converted_codes

def contract_index_clean(df):
    """
    Cleans and modifies the 'pros_index_code' column in the DataFrame based on the following rules:
    
    1. For 'pros_index_code' values that start with 'P' (but not 'PAP'):
       - Keep only the first two words.
       
    2. For 'index_code' values that start with 'F':
       - Replace the corresponding 'pros_index_code' with 'Fixed Price'.
       
    3. For 'index_code' values that start with 'R' or 'G' (but not 'GB' or 'GBR'):
       - Replace the corresponding 'pros_index_code' with the value from 'index_code_desc'.
       
    4. Strip any leading or trailing spaces from the 'pros_index_code' values.
    """
    
    # 1. Handle 'pros_index_code' values starting with 'P' (but not 'PAP')
    condition_p = (df.pros_index_code.str.startswith('P')) & (~df.pros_index_code.str.startswith('PAP'))
    df.loc[condition_p, 'pros_index_code'] = df.loc[condition_p, 'pros_index_code'].apply(lambda x: ' '.join(x.split(' ')[:2]))
    
    # 2. Replace 'pros_index_code' with 'Fixed Price' where 'index_code' starts with 'F'
    df.loc[df.index_code.str.startswith('F'), 'pros_index_code'] = 'Fixed Price'
    
    # 3. Replace 'pros_index_code' with 'index_code_desc' for 'R' and 'G' (excluding 'GB' and 'GBR')
    condition_rg = df.index_code.str.startswith(('R', 'G')) & (~df.index_code.str.startswith(('GB', 'GBR')))
    df.loc[condition_rg, 'pros_index_code'] = df.loc[condition_rg, 'index_code_desc']
    
    # 4. Strip any leading or trailing spaces from 'pros_index_code'
    df['pros_index_code'] = df['pros_index_code'].str.strip()
    
    return df


def check_word_counts_pros_index_code(string):
    """
    This checks if there are any words in the string that repeat.
    If any word repeats, it returns True, otherwise False. The tring is pros_index_code
    """
    words=string.split(' ')
    word_counts=Counter(words)
    for count in word_count.values():
        if count>1:
            return True
    return False

def updated_index_code_without_matches(df):
    """
    Based on check_word_counts_pros_index_code(string) function we will update the pros_index_code values accordingly
    """
    
    df['repeated_index_code']=df['pros_index_code'].apply(check_word_counts_pros_index_code)
    
    condition=(df['repeated_index_code']==True)

    df.loc[condition,
           'pros_index_code']=df.loc[condition,'index_code_desc']
    
    return df.drop(columns=['repeated_index_code']) 
 

###### Both contract and band price df data functions 
######
######

def both_contract_band_df_merge(contract,band):
    """
    create the df that combines both contracted price and band_price and keep those not null values 
    
    
    """
    contract_cols=['delivery_date','sales_document','qty_in_m3','qty_in_usg','qty_in_litres','qty_in_hl','document_currency',
              'grn','sector','customer_name','customer_external_lens','location_country','tender','location_desc','delivery_point','delivery_point_desc'
        ,'product_code', 'index_code' ,'index_code_desc' , 'currency_code','uom_code' ]

    band_price_cols=['delivery_date','sales_document','qty_in_m3','qty_in_usg','qty_in_litres','qty_in_hl','document_currency',
                  'grn','sector','customer_name','customer_external_lens','location_country','band_price_desc','location_desc','delivery_point','delivery_point_desc',
                     'product_code','currency_code','uom_code']
    
    df1=band[band_price_cols]
    df2=contract[contract_cols]
    
    join_left_on=['grn','sector','location_country','product_code']
    join_right_on=['grn_band_price','sector_band_price','location_country_band_price','product_code_band_price']
    
    df1.rename(columns=lambda x: x+'_band_price',inplace=True)
    
    merge_df=df2.merge(df1,left_on=join_left_on,right_on=join_right_on,how='left')
    
    merge_df=merge_df.loc[merge_df.grn_band_price.notnull()]
    return merge_df


def clean_duplicates_both_contract_band_price_df(merge_df):
    """
    for each contract price data entries we will have muptiple band_price data entries, in order to prevent double counting of the trx volumes,
    we need to only keep the 1st contract price volume history and make he rest duplicates value to zero and make sure for those trx records
    with 0 volume we would still keep the trx record's unique delivery number so that it is clear that those are the same contract price trx records
    
    
    
    """
    contract_cols=merge_df.columns.tolist()[:21]

    band_cols=merge_df.columns.tolist()[21:]

    merge_contract_df=merge_df[contract_cols]

    merge_band_df=merge_df[band_cols]
    
    clean_contract=merge_contract_df.mask(merge_contract_df.stack().duplicated().unstack(),0)
    
    new_merge_df=pd.concat([clean_contract,merge_band_df],axis=1)
    
    new_merge_df.delivery_date=merge_df.delivery_date

    new_merge_df.sales_document=merge_df.sales_document
    
    return new_merge_df

###### Data Visualization Functions Section 
###### Data Visualization
###### Data Visualization
######


                                    
def create_resample_graph_qty(df):
    
    """
    Resample final_df and create daily, monthly and quarterly qty plots
    
    
    """
    qty_type=df['qty_mapping'].value_counts().index[0]
    df['start_date']=pd.to_datetime(df['start_date'],format='%Y-%m-%d')
    qty_df=df[['start_date',qty_type]].set_index('start_date')
    sns.set_style('darkgrid')
    fig, ax = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(8,15))

    sns.lineplot(data=qty_df, ax=ax[0],markers=True)
    ax[0].set_title(f'Daily QTY in {qty_type}', fontsize=14)
    plt.xlabel('Date',fontsize=12)

    resampled_df = qty_df.resample('M').sum()
    sns.lineplot(data=resampled_df, ax=ax[1],markers=True)
    ax[1].set_title(f'Monthly Total QTY {qty_type}', fontsize=14)

    resampled_df = qty_df.resample('Q').sum()
    sns.lineplot(data=resampled_df, ax=ax[2],markers=True)
    ax[2].set_title(f'Quarterly Total QTY {qty_type}', fontsize=14)
    return plt.show()

def generate_tmp_by_date(df,day_col):
    
    tmp=df[['band_price','band_price_cat','currency_code','qty_mapping','start_date',
          'delivery_date','year_month','month','year','month_number','net_price','qty_in_m3','qty_in_usg','qty_in_litres','qty_in_hl']]
    tmp_agg=tmp.groupby([day_col,'currency_code','qty_mapping'],as_index=False).agg({'net_price':'median','qty_in_m3':'sum','qty_in_usg':'sum',
                                                                              'qty_in_litres':'sum','qty_in_hl':'sum'}).sort_values(by=[day_col])
    return tmp_agg
    

def date_feature_creation(df,day_col):
    """
    Create new aggregated df which contains all aspects of data columns 
    dataframe is solely USDC and qty_in_usg
    
    """
    
    df[day_col]=pd.to_datetime(df[day_col])
    df['year']=df[day_col].dt.year
    df['month']=df[day_col].dt.strftime('%B')
    df['month_number']=df[day_col].dt.month
    df['weekofyear']=df[day_col].dt.weekofyear
    df['day']=df[day_col].dt.day
    df['dayofweek']=df[day_col].dt.day_name()
    return df 
    

def price_median_overall_plots(df,groupby_1,groupby_2,variable,title):
    
    """
    Overall stacked price by extra lenses plots, the aggregation function is by taking the median
    
    """
    df_sales_org_median_price=df[[groupby_1,groupby_2,variable]].groupby(
        [groupby_1,groupby_2],as_index=False)[variable].median().sort_values(by=variable,ascending=False)
    sales_catarray=df_sales_org_median_price.groupby([groupby_1])[variable].sum().sort_values(ascending=False).index.to_list()
    return create_stacked_bar(df_sales_org_median_price,groupby_1, variable, groupby_2,title,sales_catarray)


def qty_sum_overall_plots(df,groupby_1,groupby_2,title):
    """
    Overall stacked qty by extra lenses plots, the aggregation function is by taking the sum
    
    """
    variable=df['qty_mapping'].mode()[0]
    df_sales_org_median_price=df[[groupby_1,groupby_2,variable]].groupby([groupby_1,groupby_2],
                                                                         as_index=False)[variable].sum().sort_values(by=variable,ascending=False)
    sales_catarray=df_sales_org_median_price.groupby([groupby_1])[variable].sum().sort_values(ascending=False).index.to_list()
    return create_stacked_bar(df_sales_org_median_price,groupby_1, variable, groupby_2,title,sales_catarray)
                                                                         
                                                                         

def create_currency_qty_dict(df):
    top_currency_qty_combo=df[['currency_code','qty_mapping']].value_counts().index.tolist()
    top_currency_qty_combo=pd.DataFrame(top_currency_qty_combo).rename(columns={0:'currency',1:'qty_mapping'})

    top_currency_qty_combo=dict(zip(top_currency_qty_combo.currency,top_currency_qty_combo.qty_mapping))
    return top_currency_qty_combo


def generate_df(df,currency,qty_type):
    plot_df=df.loc[(df['currency_code']==currency)&(df['qty_mapping']==qty_type)]
    return plot_df 



def generate_plot_qty_currency_df(df):
    """
    Generate df based on top 4 currency and qty combination 
    
    """
    dict_of_df={}
    top_combo=create_currency_qty_dict(df)
    for k, v in top_combo.items():
        print(k,v)
        key_name='df_'+str(k)
        qty_type=top_combo.get(k)
        dict_of_df[key_name]=generate_df(df,k,v)
    return dict_of_df


def print_qty_currency_df_length_message(df,dict_df):
    total_len=0
    for k in dict_df.keys():
        total_len+=dict_df[k].shape[0]
    return print(f'Total Len of dict_df: {total_len} \n' 
          f'Total Len of df: {df.shape[0]}\n'
          f'Percent of df: {round(total_len/df.shape[0],2)}'
         )



def create_scatter_plot_price_qty(df,start_range=None):
    scatter_tmp=df.loc[(df.net_price >0) &(df.qty_in_usg>0)]
    scatter_tmp=scatter_tmp.loc[(scatter_tmp.net_price<5000)]
    currency_value=df['currency_code'].value_counts().index[0]
    qty_value=df['qty_mapping'].value_counts().index[0]
    fig=px.scatter(scatter_tmp,x='net_price',y=qty_value,
                   size=qty_value,size_max=30,color=qty_value,log_y=True)

    fig.update_layout({'title':f'Correlation between Net Price {currency_value} and Qty {qty_value} excluding outliers'})
    start=1 if start_range is None else start_range
    fig.update_yaxes(range = [start,5])
    return fig.show('iframe')

def create_log_qty_by_qty_type(df):
    """
    Create log y volume box plot based on qty_mapping value
    df is using dict_df based on top_3currency_qty_combo
    
    """
    qty_type=df['qty_mapping'].value_counts().index[0]
    log_qty=df[['year','month_number',qty_type]]
    log_qty[f'log_{qty_type}']=np.log(log_qty[qty_type])
    name=f'log_{qty_type}'
    variables=name
    return plot_box_seasonality(log_qty,'year',variables,'month_number',base_value=qty_type)

def create_two_y_axis_price_qty_plot(df,y1):
    """
    This plot has two y axis which can be used to plot two y dimension in one plot 
    
    """
    currency=df['currency_code'].value_counts().index[0]
    qty_type=df['qty_mapping'].value_counts().index[0]
    plot_df=df.loc[(df['currency_code']==currency) &(df['qty_mapping']==qty_type)]
    fig=make_subplots(specs=[[{'secondary_y':True}]])
    fig.add_trace(
    go.Bar(
        x=plot_df['year_month'],
        y=plot_df[qty_type],
        name=f'Transactional QTY {qty_type}',
        hoverinfo='none' ,               #Hide the hoverinfo
        marker_color='#d99b16'

    ),
    secondary_y=False
    )
    fig.add_trace(
    go.Scatter(
        x=plot_df['year_month'],
        y=plot_df[y1],
        name=f'Median {y1} in {currency}',
        mode='lines',
        text=plot_df[y1],
        hoverinfo='text',
        line = dict(color='#f70f13', width=3)
    ),
    secondary_y=True
    )
    fig.update_layout(hoverlabel_bgcolor='#DAEEED',  #Change the background color of the tooltip to light blue
                 title_text=f"Median Value Distribution of {y1} in {currency} & Total Volume in {qty_type}", #Add a chart title
                 title_font_family="Times New Roman",
                 title_font_size = 20,
                 title_font_color="darkblue", #Specify font color of the title
                 title_x=0.46, #Specify the title position
                 xaxis=dict(
                        tickfont_size=10,
                        tickangle = 270,
                        showgrid = True,
                        zeroline = True,
                        showline = True,
                        #showticklabels = True,
                        #dtick="M1", #Change the x-axis ticks to be monthly
                        tickformat="%b\n%Y"
                        ),
                 legend = dict(orientation = 'h', xanchor = "center", x = 0.45, y= 1.11), #Adjust the legend position
                 yaxis_title=f'Transactional Total Volume {qty_type}',
                 yaxis2_title=f'Median {y1} in {currency}')


    return fig.show('iframe')

def get_top10_country(df):
    top10_list=df['country_name'].value_counts()[:10].index.tolist()
    return top10_list

def price_comparison_plot(df,groupby_col1,groupby_col2,qty_list_no):
    
    if df['currency_code'].nunique() ==1:
        currency_value=df['currency_code'].mode()[0]
    else:
        currency_value='all currencies'
    
    qty_lists=['qty_in_usg', 'qty_in_litres', 'qty_in_hl', 'qty_in_m3']
    price=df.groupby([groupby_col1,groupby_col2],as_index=False)['net_price'].median()
    fig=px.line(price,x=groupby_col1,y='net_price',markers='circle',color=groupby_col2)
    fig.update_layout({f'title':f'Median Net Price in {currency_value} per {qty_lists[qty_list_no]} Overview by {groupby_col2}'})
    return fig.show('iframe')

def create_price_comparison_plots_by_qty_type(df,qty_list_no):
    """
    create plots iterate through qty_lists to plot every qty type price comparison for all currencies 
    
    
    """
    qty_lists=df.qty_mapping.unique().tolist()
    
    return price_comparison_plot(df.loc[df['qty_mapping']==qty_lists[qty_list_no]],'year_month','currency_code',qty_list_no)
  
def create_euro5_price_comparison(df,qty_list_no):
    """
    create EURO5 price comparison by countries 
    
    """
    qty_lists=df.qty_mapping.unique().tolist()
    eur5_df=df.loc[(df['currency_code']=='EUR5')&(df['qty_mapping']==qty_lists[qty_list_no])]
    return price_comparison_plot(eur5_df,'year_month','country_name',qty_list_no)

def create_seasonal_line_by_year_plot(df,variables):
    """
    Create seasonal line plot by year 
    df is by combo of currency & qty type
    
    """
    fig,ax=plt.subplots(figsize=(15,6))
    df=df.sort_values(by=['month_number'])
    sns.lineplot(x=df['month'],y=df[variables],hue=df['year'])
    currency_value=df['currency_code'].mode()[0]
    qty_type=df['qty_mapping'].mode()[0]
    ax.set_title(f'Seasonal plot of Net Price in {currency_value} measured in {qty_type}',fontsize=20,loc='center',fontdict=dict(weight='bold'))
    ax.set_xlabel('Month',fontsize=12,fontdict=dict(weight='bold'))
    ax.set_ylabel('Net Price',fontsize=12,fontdict=dict(weight='bold'))
    return plt.show()
                                                                         
def plot_volume_by_currency_qty_year(df):
    """
    plot seasonal YOY plots for qty
    
    """
    currency_value=df['currency_code'].mode()[0]
    qty_type=df['qty_mapping'].mode()[0]
    tmp=df.groupby(['month','year'],as_index=False)[qty_type].sum()
    fig=px.bar(tmp,x='month',y=qty_type,color='year')
    cat=df[['month','month_number']].drop_duplicates().sort_values('month_number').month.tolist()
    fig.update_xaxes(categoryorder='array', categoryarray= cat)
    fig.update_layout({'title':f'Distribution of Total Volume in {qty_type} for {currency_value} currency'})
    return fig.show('iframe')


                                                                        

def create_rev_pie_band_price_breakdown_plots(df,rev_list_no=None):
    
    """
    Creating revenue breakdown pie plots by country and bar plots by band price cat
    """
    rev_lists=df.revenue_mapping.unique().tolist()
    currency_value=df['currency_code'].mode()[0]
    
    
    if currency_value=='EUR5':
        
        eur5_df=df.loc[(df['currency_code']=='EUR5')&(df['revenue_mapping']==rev_lists[rev_list_no])]
        rev_type=eur5_df['revenue_mapping'].mode()[0]
        
        rev_band=eur5_df.groupby(['band_price_cat','year'],as_index=False)[rev_type].sum()
        rev_country=eur5_df.groupby(['country_name'])[rev_type].sum()
        
    else: 
        rev_type=df['revenue_mapping'].mode()[0]
        rev_band=df.groupby(['band_price_cat','year'],as_index=False)[rev_type].sum()
        rev_country=df.groupby(['country_name'])[rev_type].sum()
        
    
    
    fig = make_subplots(rows=1, cols=2,  specs=[[{"type": "bar"}, {"type": "pie"}]],
                    subplot_titles=(f"Total Revenue in {currency_value} measured in {rev_type} by Band Price Type", f"Proportion of revenue in {currency_value} by country"))
    #SALES 
    fig.append_trace(go.Bar(x=rev_band['band_price_cat'], y=rev_band[rev_type],marker_color=rev_band['year'], text = rev_band['year']
                            ),
                    row=1, col=1)

    cat=rev_band.sort_values(by=[rev_type],ascending=False)['band_price_cat'].tolist()
    fig.update_xaxes(categoryorder='array', categoryarray= cat)

    fig.append_trace(go.Pie(values=rev_country.values, labels=rev_country.index,
                           ),
                     row=1, col=2)




    ##styling
    #fig.update_yaxes(showgrid=False, ticksuffix=' ', categoryorder='total ascending', row=1, col=1)
    fig.update_xaxes(visible=True, row=1, col=1)


    fig.update_layout(template="ggplot2",
                      bargap=0.4,
                      height=700,
                      width=1300,
                      showlegend=True)

    return fig.show('iframe')

def revenue_sum_overall_plots(df,groupby_1,groupby_2,title,rev_list_no=None):
    """
    Overall stacked revenue by extra lenses plots, the aggregation function is by taking the sum
    
    """
    rev_lists=df.revenue_mapping.unique().tolist()
    currency_value=df['currency_code'].mode()[0]
    if currency_value=='EUR5':
        df=df.loc[(df['currency_code']=='EUR5')&(df['revenue_mapping']==rev_lists[rev_list_no])].copy()
        variable=df['revenue_mapping'].mode()[0]
    else:
        variable=df['revenue_mapping'].mode()[0]
    
    df_sales_org_median_price=df[[groupby_1,groupby_2,variable]].groupby([groupby_1,groupby_2],
                                                                         as_index=False)[variable].sum().sort_values(by=variable,ascending=False)
    sales_catarray=df_sales_org_median_price.groupby([groupby_1])[variable].sum().sort_values(ascending=False).index.to_list()
    return create_stacked_bar(df_sales_org_median_price,groupby_1, variable, groupby_2,title,sales_catarray)

def create_cost_breakdown_line_plot(df,cost_cols,qty_list_no=None):
    """
    Cost breakdown plots by timeline 
    
    
    """
    currency_value=df['currency_code'].mode()[0]
    qty_type=df['qty_mapping'].mode()[0]
    lc_cost_cols=[c for c in cost_cols if 'lc' in c]
    if df['currency_code'].mode()[0]=='EUR5':
        qty_lists=['qty_in_litres', 'qty_in_m3', 'qty_in_hl']
        df=df.loc[df['qty_mapping']==qty_lists[qty_list_no]].copy()
        tmp=df.groupby(['year_month'],as_index=False)[lc_cost_cols].sum()
        qty_type=qty_lists[qty_list_no]
    else:
        tmp=df.groupby(['year_month'],as_index=False)[lc_cost_cols].sum()
        
    fig=px.line(tmp,'year_month',tmp.columns[1:],markers='circle')
    fig.update_layout({'title':f'Total Cost Breakdown Overview in {currency_value} measured in {qty_type}'})
    return fig.show('iframe')

def generate_cost_per_qty_mapping(df):
    """
    Generate cost per unit columns by matching it with its respective qty type
    
    
    """
    qty_columns=[c for c in df.columns if c.startswith('qty')][1:]
    for c in qty_columns:
        name=f'cost_per_{c}_lc'
        df[name]=round(df['l3_cost_lc']/df[c])

    cost_mapping={'qty_in_usg':'cost_per_qty_in_usg_lc',

                'qty_in_hl':'cost_per_qty_in_hl_lc',
               'qty_in_m3' :'cost_per_qty_in_m3_lc',
               'qty_in_litres' :'cost_per_qty_in_litres_lc'}
    df['cost_mapping']=df['qty_mapping'].map(cost_mapping)
    df.fillna(0,inplace=True)
    
    return df


#### PRICE ELASTICITY TRANSFORMATION SECTION 
####
####
####
####
def sort_df(df,date_col):
    """
    Sort the df in ascending order"
    
    """
    df[date_col]=pd.to_datetime(df[date_col],format='%Y-%m-%d')
    df.sort_values(by=[date_col],inplace=True)
    df=df.reset_index(drop=True)
    
    return df 


def date_features(df,date_col, label=None):
    
    """
    In order to factor in seasonality trend in Prophet model, we would need to have a look at the data's date features 
    and see if we could find any consistent trend and then we could take away those patterns in the modelling step
    
    """
    df = df.copy()

    df['date'] = pd.to_datetime(df[date_col])
    df['month'] = df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.strftime('%Y')
    df['dayofweek'] = df['date'].dt.strftime('%A')
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

def plot_date_feature_seasonality(df_new,y_label):
    """
    Create plots to showcase seasonality trends of data
    
    """
    fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)
    fig.set_size_inches(20,30)

    monthAggregated = pd.DataFrame(df_new.groupby("month")[y_label].sum()).reset_index().sort_values(y_label)
    sns.barplot(data=monthAggregated,x="month",y=y_label,ax=ax1)
    ax1.set(xlabel='Month', ylabel='Total Sales received')
    ax1.set_title("Total Sales received By Month",fontsize=15)

    monthAggregated = pd.DataFrame(df_new.groupby("dayofweek")[y_label].sum()).reset_index().sort_values(y_label)
    sns.barplot(data=monthAggregated,x="dayofweek",y=y_label,ax=ax2)
    ax2.set(xlabel='dayofweek', ylabel='Total Sales received')
    ax2.set_title("Total Sales received By Weekday",fontsize=15)

    monthAggregated = pd.DataFrame(df_new.groupby("quarter")[y_label].sum()).reset_index().sort_values(y_label)
    sns.barplot(data=monthAggregated,x="quarter",y=y_label,ax=ax3)
    ax3.set(xlabel='Quarter', ylabel='Total Sales received')
    ax3.set_title("Total Sales received By Quarter",fontsize=15)

    monthAggregated = pd.DataFrame(df_new.groupby("year")[y_label].sum()).reset_index().sort_values(y_label)
    sns.barplot(data=monthAggregated,x="year",y=y_label,ax=ax4)
    ax4.set(xlabel='year', ylabel='Total Sales received')
    ax4.set_title("Total Sales received By year",fontsize=15)
    
    return plt.show()


def create_prophet_x_train(df_agg,end_date):
    """
    create Prophet x_train df for model training to create baseline log sales
    
    """
    df_agg=sort_df(df_agg,'delivery_date')

    ## seasonality trend: 3rd quarters high volume,July to Sep, weekend low volume, 
    df=df_agg[['delivery_date','ln_qty_in_usg']].copy()
    df.rename(columns={'delivery_date':'ds','ln_qty_in_usg':'y'},inplace=True)

    df_grouped=df.groupby(['ds'],as_index=False)['y'].median()

    mask1=(df_grouped['ds']<end_date)
    mask2=(df_grouped['ds']>=end_date)

    X_train=df_grouped.loc[mask1]
    X_test=df_grouped.loc[mask2]
    print(X_train.shape,X_test.shape)
    return X_train,X_test,df_grouped

def prophet_hyperparameter_tuning(df):
    
    params_grid={'seasonality_mode':('multiplicative','additive'),
             'changepoint_prior_scale':[0.1,0.2,0.3,0.4], # indicates how flexible the change point are allowed to be, 
                                                            # change points means the number of changes happen in the data. 
                                                            # The higher the more flexibility, default is 0.05, but will end up overfitting
            'holidays_prior_scale':[0.1,0.2,0.3,0.4],
              'n_changepoints' : [50,100,150]
             }

    grid=ParameterGrid(params_grid)
    cnt=0
    for p in grid:
        cnt+=1
    
    strt='2020-08-01'
    end='2024-02-28'
    model_parameters=pd.DataFrame(columns=['MSE','Parameters'])
    
    for p in grid:
        test=pd.DataFrame()
        print(p)
        random.seed(42)
        train_model=Prophet(changepoint_prior_scale=p['changepoint_prior_scale'],
                holidays_prior_scale=p['holidays_prior_scale'],
                n_changepoints=p['n_changepoints'],
                seasonality_mode=p['seasonality_mode'],
                weekly_seasonality=True,
                daily_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.95 )
        train_model.fit(X_train)
        future=train_model.make_future_dataframe(periods=59,freq='D')
        train_forecast=train_model.predict(future)
        test=train_forecast[['ds','yhat']]
        actual=df.loc[(df.ds>=strt)&(df.ds<=end)]

        mse=mean_squared_error(actual['y'],test['yhat'])
        print(f'MSE:{mse}')
        model_parameters=model_parameters.append({'MSE':mse,'Parameters':p},ignore_index=True)
        
    return model_parameters






def generate_log_qty(df):
    """
    in order to fit in constant price elasticity equation, we need to generate log qty values for each qty type
    
    
    """
    qty_cols=['qty_in_m3','qty_in_litres','qty_in_hl','qty_in_usg']
    for c in qty_cols:
        name=f'ln_{c}'
        df[name]=np.log2(df[c],out=np.zeros_like(df[c]),where=(df[c]!=0))
    return df 

def fb_trend_level_component(df,timestamp_var,baseline_dep_var,changepoint_prior_scale_value=None):
    """
    Using FB Prophet to generate trend log unit sales, the log has to be calucated at per price and start_date aggregated level 
    
    """
    # Preparing the datasecloset
    
    df=df[[timestamp_var,baseline_dep_var,'AVGAS_USDC_USG','JET_CIF_USDC_USG']]
    df=df.sort_values(by=['start_date']).reset_index(drop=True)
    df = df.rename(columns={timestamp_var: 'ds', baseline_dep_var: 'y'})
    
    df['ds'] = pd.to_datetime(df['ds'])
    
        # Initializing and fitting the model
    model = Prophet(changepoint_prior_scale= changepoint_prior_scale_value) #Default changepoint_prior_scale = 0.05
    
    # add a list of external regressors to factor in COVID effect 
    regressors = ['AVGAS_USDC_USG','JET_CIF_USDC_USG']
    for regressor in regressors:
                model = model.add_regressor(regressor,
                                                      prior_scale=1,  
                                                     standardize='auto',   mode='multiplicative')

    model.fit(df)
    # Since we are only decomposing current time series, we will use same data is forecasting that was used for modelling
    # Making predictions and extracting the level component
    forecast=model.predict(df)
    forecast['ds']=pd.to_datetime(forecast['ds'])
    level_component=forecast['trend']
    df=pd.concat([df,forecast[['trend']]],axis=1)
    name=f'trend_{changepoint_prior_scale_value}'
    df=df.rename(columns={'trend':name})
    
    return df

def fb_create_trend_comparison_line_plots_sum(df):
    tmp=df[['start_date','ln_qty_in_usg','trend_0.3','trend_0.9']]
    tmp_agg=tmp.groupby(['start_date'],as_index=False)[['ln_qty_in_usg','trend_0.3','trend_0.9']].sum()
    fig=px.line(tmp_agg,x='start_date',y=['ln_qty_in_usg','trend_0.3','trend_0.9'],markers='circle')
    fig.update_layout({'title':'SUM Trend Component at different values of changepoint_prior_scale'})
    return fig.show('iframe')

def fb_create_trend_comparison_line_plots_median(df):
    tmp=df[['start_date','ln_qty_in_usg','trend_0.3','trend_0.9']]
    tmp_agg=tmp.groupby(['start_date'],as_index=False)[['ln_qty_in_usg','trend_0.3','trend_0.9']].median()
    fig=px.line(tmp_agg,x='start_date',y=['ln_qty_in_usg','trend_0.3','trend_0.9'],markers='circle')
    fig.update_layout({'title':'Median Trend Component at different values of changepoint_prior_scale'})
    return fig.show('iframe')

def fb_generate_different_levels_trend_component(df1,df2,original_df):
    """
    generate df with two levels of change_point values and pick the smoother one as log base unit sales 
    
    """
    df=pd.concat([df1,df2.iloc[:,-1]],axis=1)
    original_df=original_df.sort_values(by=['start_date']).reset_index(drop=True)

    original_df['trend_0.3']=df.iloc[:,-2]

    original_df['trend_0.9']=df.iloc[:,-1]
    return original_df

def preparing_price_elasticity_matrix(df):
    """
    Preparing for the Optimization matrix to feed into the optimization algorithm 
    """
    df_model=df[['start_date','net_price','trend_0.3']]
    df_model['ln_net_price']=np.log2(df_model['net_price'])
    df_model=df_model.rename(columns={'trend_0.3':'ln_base_sales'})
    df_model=df_model[['start_date','ln_net_price','ln_base_sales']]
    
    # Preparing the matrix to feed into optimization algorithm
    x=df_model
    x['intercept']=1
    x=x[['intercept','ln_net_price','ln_base_sales']].values.T
    actuals=x[2]
    return df_model,x, actuals

#define the objective function to be minimized,in our case, it will be the loss function that we use in linear regression MSE (pred-actual => [intercept + elasticity*ln_price — actual]²) 
def objective(x0):
    return sum(((x[0]*x0[0]+x[1]*x0[1])-actuals)**2) #  MSE (pred-actual => [intercept + elasticity*ln_price — actual]²) 

def generate_price_elasticity_per_df(df,x,actuals,pe_lower_bound=None,pe_upper_bound=None,method=None):
    #define the inital guess for parameters of intercept and elasticity
    x0=[1,-1]
    #define the bounds for the parameters
    bounds=((None,None),(pe_lower_bound,pe_upper_bound))
    # Use the SLSQP optimization algorithm to minimize the objective function
    result=minimize(objective,x0,bounds=bounds,method='L-BFGS-B')
    print(result)
    price_elasticity=result.x[1]
    df['price_elasticity']=result.x[1]
    return df 

def fb_model_output_plot(df,timestamp_var,baseline_dep_var,changepoint_prior_scale_value=None):
    """
    Using FB Prophet to generate trend log unit sales, the log has to be calucated at per price and start_date aggregated level 
    
    """
    # Preparing the datasecloset
    
    df=df[[timestamp_var,baseline_dep_var,'AVGAS_USDC_USG',
       'JET_CIF_USDC_USG']]
    df=df.sort_values(by=['start_date']).reset_index(drop=True)
    df = df.rename(columns={timestamp_var: 'ds', baseline_dep_var: 'y'})
    
    df['ds'] = pd.to_datetime(df['ds'])
    
        # Initializing and fitting the model
    model = Prophet(changepoint_prior_scale= changepoint_prior_scale_value) #Default changepoint_prior_scale = 0.05
    
    #add a list of external regressors to factor in COVID effect 
    regressors = ['AVGAS_USDC_USG',
       'JET_CIF_USDC_USG']
    for regressor in regressors:
                model = model.add_regressor(regressor,
                                                      prior_scale=1,  
                                                     standardize='auto',   mode='multiplicative')
    

    model.fit(df)
    # Since we are only decomposing current time series, we will use same data is forecasting that was used for modelling
    # Making predictions and extracting the level component
    forecast=model.predict(df)
    return model.plot_components(forecast)

def convert_df_regressor(df,df2,cols=None):
    """
    Include Consumer Confidence Index as regressor for Prophet model
    
    
    """
    df['year_month']=df['start_date'].apply(lambda x:pd.to_datetime(x).strftime('%Y-%m'))
    
    fts=df2.columns if cols is None else cols

    df_merge=df.merge(df2[fts],left_on='year_month',right_on='year_month',how='left').drop(['year_month'],axis=1)

    return df_merge


def process_commodity_index(df):
    """
    Preprocess commdity index df and convert USD/Tonne to USDC/USG for USDC model df 
    
    """
    df=df[['Date','GSLN Prem Unl10ppms Fob Rdam Brg (D-1) USD/Tonne','Jet CIF NWE Cargo USD/Tonne']].reset_index(drop=True)

    df['Jet CIF NWE Cargo USD/Tonne']=df['Jet CIF NWE Cargo USD/Tonne'].apply(lambda x: x*330.215)
    df['GSLN Prem Unl10ppms Fob Rdam Brg (D-1) USD/Tonne']=df['GSLN Prem Unl10ppms Fob Rdam Brg (D-1) USD/Tonne'].apply(lambda x:x*369.472*1.6)

    df=df.rename(columns={'GSLN Prem Unl10ppms Fob Rdam Brg (D-1) USD/Tonne':'AVGAS_USDC_USG',
                              'Jet CIF NWE Cargo USD/Tonne':'JET_CIF_USDC_USG'})
    df['year_month']=df['Date'].apply(lambda x:x.strftime('%Y-%m'))
    
    median_df=df.groupby(['year_month'],as_index=False)['AVGAS_USDC_USG','JET_CIF_USDC_USG'].median()
    return df,median_df

def plot_com_index_lines(df,date,col1,col2):
    """
    Line plots for commodity index 
    
    """
    fig=px.line(df,date,[col1,col2])
    fig.update_layout({'title':'Commodity Platts Index for Jet & Avgas'})
    return fig.show('iframe')

def transform_standard_scaler(df,col):
    """
    Use StandardScaler to standardise col for commodity index cols 
    
    """
    scaler=StandardScaler()

    array1=df[col].to_numpy().reshape(-1, 1)

    array1_transformed=scaler.fit_transform(array1)

    my_list = map(lambda x: x[0], array1_transformed)

    my_list=pd.Series(my_list)
    return my_list
