import pandas as pd
import numpy as np
import pydata_google_auth
from fastcore.all import *
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import roc_auc_score,precision_recall_curve,f1_score,auc,plot_confusion_matrix,confusion_matrix,plot_roc_curve 
# %matplotlib inline 

import seaborn as sns

plt.style.use('fivethirtyeight')
five_thrity_eight=[
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",    
]

def download_data(project_id,auth= None, sign= None, time_constraint = None,query =None):
    """Downlaod and authenticate the data"""
    if project_id is None: raise ValueError('Argument cannot be none')
    auth = pydata_google_auth.get_user_credentials(scopes=["https://www.googleapis.com/auth/bigquery"]) 
    query = query 
    if time_constraint is not None: query +=f"WHERE started_at {sign}{time_constraint}"
    df=pd.read_gbq(query=query, project_id=project_id,credentials=auth, 
                 progress_bar_type="tqdm_notebook", use_bqstorage_api=True)
    return df

class Preprocessing:
    
    def __init__(self,df):
        self.df = df
    
    def clean_df(self):
        self.df=self.df.loc[self.df['is_active']==1]
        self.df=self.df.drop(['started_at','is_active','date'],axis=1)
    
    def check_null(self):
        flag=self.df.isnull().sum().any()
        if flag == True:
            total=self.df.isnull().sum().sort_values(ascending=False)
            percent=(self.df.isnull().sum()/self.df.isnull().count()).sort_values(ascending=False)
            missing=pd.concat([total,percent],axis=1,keys=['total','percent'])
            
            data_type=[]
            for col in self.df.columns:
                dtypes=self.df[col].dtype
                data_type.append(dtypes)
            missing['dtype']=data_type
            return missing
        else:
            return False
    
    def fill_null(self):
        num_columns=self.df.select_dtypes(include=np.number).columns
        self.df=self.df.fillna(0)  ## Fill apps data with 0 for no activities
        
    def split_answers(self,index):
        self.data=self.df.pivot(index=index,columns='question_id',
                      values='choice_ids').reset_index()
        for answer in ['DEBT', 'EDUCATION', 'HOLIDAY', 'INCOME', 'OTHER']:
            self.data[f'SPEND_MONTHLY_INCOME_ANSWER_{answer}']=self.data.HOW_TO_SPEND_MONTHLY_INCOME.str.contains(answer)
        for col in ['SPEND_MONTHLY_INCOME_ANSWER_DEBT',
       'SPEND_MONTHLY_INCOME_ANSWER_EDUCATION',
       'SPEND_MONTHLY_INCOME_ANSWER_HOLIDAY',
       'SPEND_MONTHLY_INCOME_ANSWER_INCOME',
       'SPEND_MONTHLY_INCOME_ANSWER_OTHER']:
            le=LabelEncoder()
            self.data[col]=le.fit_transform(self.data[col])
        self.data=self.data.drop(['HOW_TO_SPEND_MONTHLY_INCOME'],axis=1)
                   
        self.data['total_spend_monthly_income_answer_count']=self.data['SPEND_MONTHLY_INCOME_ANSWER_DEBT']+            self.data['SPEND_MONTHLY_INCOME_ANSWER_EDUCATION']+ self.data['SPEND_MONTHLY_INCOME_ANSWER_HOLIDAY']+ self.data['SPEND_MONTHLY_INCOME_ANSWER_INCOME']+ self.data['SPEND_MONTHLY_INCOME_ANSWER_OTHER']
        return self.data
    

def plot_columns(df,question_columns,group_column):
    
    for col in question_columns:
        f,ax=plt.subplots(figsize=(6,4))
        (pd.crosstab(df[group_column],df[col]).div(pd.crosstab(df[group_column],df[col]).sum(axis=1),axis=0)).plot(kind='bar',ax=ax)
        ax.set_title(col)
        ax.legend(loc='upper left')
        
def plot_all_columns_plot(df,group_name,question_name):
    f,ax=plt.subplots(figsize=(8,6))
    (df.groupby([group_name])[question_name].sum().T/df[group_name].value_counts()).plot(kind='barh',ax=ax)
    ax.set_ylabel("")

def plot_dist(df,column):
    color={0:'b',1:'g',2:'y'}
    for cluster in [0,1.0,2.0]: 
        ax = sns.displot(df.loc[df.umap_y_pred==cluster,column],kde=False,color=color.get(cluster))
        ax.fig.suptitle(f'Cluster {cluster}')
        
class Encoding:
    
    def __init__(self,df):
        self.df=df
    
    def categorise_cat(self,ordinals,cat_features):        
        cat_dict={}
        for c in cat_features:
            categories=ordinals.get(c)
            self.df[c]=pd.Categorical(self.df[c],categories=categories,ordered=categories is not None)
            cats=self.df[c].cat.categories
            cat_dict[c]=cats
            self.df[c]=self.df[c].cat.codes
        return cat_dict,self.df
    

    
def remove_outliers(df,c):
    df[f'outliers_removed_{c}']=df[c].between(df[c].quantile(.05),df[c].quantile(.95))
    return df
            
def sort_cat_label(df):
    cat_list=df['HOW_MANY_IN_HOUSEHOLD'].unique().tolist()
    cat_api=pd.api.types.CategoricalDtype(categories=cat_list,ordered=True)
    df['HOW_MANY_IN_HOUSEHOLD']=df['HOW_MANY_IN_HOUSEHOLD'].astype(cat_api)
    cat_list=['1_TO_500','500_TO_2500','2500_PLUS']
    cat_api=pd.api.types.CategoricalDtype(categories=cat_list,ordered=True)
    df.HOW_MUCH_MONTHLY_INCOME=df.HOW_MUCH_MONTHLY_INCOME.astype(cat_api)
    cat_list=['1_TO_5_HOURS','5_TO_10_HOURS','10_TO_15_HOURS','15_PLUS_HOURS']
    cat_api=pd.api.types.CategoricalDtype(categories=cat_list,ordered=True)
    df.HOW_MUCH_TIME_TO_COMMIT=df.HOW_MUCH_TIME_TO_COMMIT.astype(cat_api)
    return df  

def synthetic_clusters(embedding):
    group_0_index=np.where((embedding[:,0]>20))
    group_1_index=np.where((embedding[:,0]<20)&(embedding[:,1]<14))
    group_2_index=np.where((embedding[:,0]<20)&(embedding[:,1]>14))
    group_0=pd.DataFrame(embedding[group_0_index],columns=list('01')).set_index(np.array(group_0_index)[0])
    group_0['cluster']=0
    group_0=pd.DataFrame(group_0,columns=['0','1','cluster'])
    group_1=pd.DataFrame(embedding[group_1_index],columns=list('01')).set_index(np.array(group_1_index)[0])
    group_1['cluster']=1
    group_1=pd.DataFrame(group_1,columns=['0','1','cluster'])
    group_2=pd.DataFrame(embedding[group_2_index],columns=list('01')).set_index(np.array(group_2_index)[0])
    group_2['cluster']=2
    group_2=pd.DataFrame(group_2,columns=['0','1','cluster'])
    synthetic_clusters=group_0.append(group_1).append(group_2).sort_index()
    return synthetic_clusters     
     
def plot_acorn(df,c,group_column):
    f,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,8))
    ct = pd.crosstab(df[c],df[group_column]).drop('Unknown')
    (ct/ct.sum()).plot(kind='bar',ax=ax1)
    (ct.T/ct.T.sum()).plot(kind='bar',ax=ax2)
    ax1.set_title('Probability of catergory, given a Cluster')
    ax2.set_title('Probability of Cluster, given a category')
    ax2.legend(loc='upper left')           
        

def tenure_compute(df,cols_list,base_col):
    for c in cols_list:
        title=re.sub('_date','',c)
        df.loc[:,'tenure_'+title+'_from_'+base_col]=df[c]-df[base_col]
    return df

           
def convert_tenure_dates(df):
    tenure_cols=df.columns[df.columns.str.contains('tenure')].tolist()
    for c in tenure_cols:
        df[c]=df[c].dt.days
    return df           
        
def convert_to_datetime(df,cols):
    for c in cols:        
        df[c]=df[c].dt.date
    return df

def model_length_compute(df,target,ratio_train,ratio_int_valid):
    ldf=len(df)
    len_trn=int(df[target].sum()/ratio_train)
    len_valid=int(ldf/ratio_int_valid)
    return len_trn,len_valid

def split_df(df,ln_train,ln_val,target):
    df=df.sample(frac=1.).reset_index(drop=True)
    val=df.iloc[-ln_val:].reset_index(drop=True)
    train=df.iloc[:-ln_val].reset_index(drop=True)
    has_gathered=train[train[target]==1]
    has_not_gathered=train[train[target]==0].head(ln_train-len(has_gathered))
    tra=pd.concat([has_gathered,has_not_gathered],axis=0).sample(frac=1.).reset_index(drop=True)
    return tra,val

def split_x_y(train,valid,target_label):
    X_train,y_train=train.drop([target_label],axis=1),train[target_label].values
    X_valid,y_valid=valid.drop([target_label],axis=1),valid[target_label].values
    return X_train,y_train,X_valid,y_valid

def auc_score(model,x,y):
    rf_probs=model.predict_proba(x)
    rf_probs=rf_probs[:,1]
    predict=model.predict(x)
    rf_precision,rf_recall,_=precision_recall_curve(y,rf_probs)
    rf_f1,rf_auc=f1_score(y,predict),auc(rf_recall,rf_precision)
    return rf_f1,rf_auc

def check_parameters(parameters,values,fixed={},features=None):
    scores=[]
    f1=[]
    auc=[]
    for p in values:
        print(f'Fitting with {parameters}={p}')
        fts= X_train.columns if features is None else features
        kw = {parameters:p, **fixed}
        model=RandomForestClassifier(**kw)
        model.fit(X_train[fts],y_train)
        s=roc_auc_score(y_valid,model.predict_proba(X_valid)[:,1])
        rf_f1,rf_auc=auc_score(model,X_valid[fts],y_valid)
        
        print('ROC AUC Score',s)
        print('F1',rf_f1)
        print('Auc',rf_auc)
        print('')
        scores.append(rf_auc)
        f1.append(rf_f1)
        auc.append(rf_auc)
    plt.title(parameters)
    plt.plot(values,scores)

def print_report(model,train,valid,target_label,feats=None):
    X_train,y_train=train.drop([target_label],axis=1),train[target_label].values
    X_valid,y_valid=valid.drop([target_label],axis=1),valid[target_label].values
    
    if feats is not None:
        X_train,X_valid = X_train[feats],X_valid[feats]
        
    train_pred,val_pred=model.predict(X_train),model.predict(X_valid)
    roc_auc_train=roc_auc_score(y_train,model.predict_proba(X_train)[:,1])
    roc_auc_val = roc_auc_score(y_valid,model.predict_proba(X_valid)[:,1])
    train_f1,train_auc=auc_score(model,X_train,y_train)
    val_f1,val_auc=auc_score(model,X_valid,y_valid)
        
    res =f"""
    Training ROC_AUC :{roc_auc_train}
    Training F1 : {train_f1}
    Training AUC: {train_auc}

    Validation ROC_AUC: {roc_auc_val}
    Validation F1: {val_f1}
    Validation AUC: {val_auc}

    """
    print(res)

    plot_roc_curve(model,X_valid,y_valid)
    plot_confusion_matrix(model,X_valid,y_valid)
    tn,fp,fn,tp=confusion_matrix(y_valid,val_pred).ravel()
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    print('True Negatives:',tn)
    print('False Positives:',fp)
    print('False Negatives',fn)
    print('True Positives:',tp)
    print('Recall:',recall)
    print('Precision:',precision)

def print_lgb_report(model,train,valid,feats=None): 
    
    if feats is not None:
        X_train,X_valid = train.data[feats],valid.data[feats]
    else:
        X_train,X_valid = train.data,valid.data
    y_train,y_valid=train.label,valid.label
        
    train_pred,val_pred=model.predict(X_train),model.predict(X_valid)
    roc_auc_train=roc_auc_score(y_train,model.predict_proba(X_train)[:,1])
    roc_auc_val = roc_auc_score(y_valid,model.predict_proba(X_valid)[:,1])
    train_f1,train_auc=auc_score(model,X_train,y_train)
    val_f1,val_auc=auc_score(model,X_valid,y_valid)
        
    res =f"""
    Training ROC_AUC :{roc_auc_train}
    Training F1 : {train_f1}
    Training AUC: {train_auc}

    Validation ROC_AUC: {roc_auc_val}
    Validation F1: {val_f1}
    Validation AUC: {val_auc}

    """
    print(res)

    plot_roc_curve(model,X_valid,y_valid)
    plot_confusion_matrix(model,X_valid,y_valid)
    tn,fp,fn,tp=confusion_matrix(y_valid,val_pred).ravel()
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)

    print('True Negatives:',tn)
    print('False Positives:',fp)
    print('False Negatives',fn)
    print('True Positives:',tp)
    print('Recall:',recall)
    print('Precision:',precision)

def quality_control_date_diff_less_0(df,exception_label):
    for c in [col for col in df.columns if 'tenure' in col]:
        if df[c].min()<0 and c!=exception_label:
            print(c)
            df.drop((df[df[c]<0]).index,axis=0,inplace=True)
    return df 

def pass_df(q):
    if q==power_up_sql:
        df=download_data(project_id='uw-data-warehouse-prod',query=power_up_sql)
    else:
        df=download_data(project_id='uw-data-warehouse-prod',query=sql)
    return df