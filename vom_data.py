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



class Preprocessing:
    """
    A class to cleanse and check null for dataframe
    
    Attributes
    ----------
    df : dataframe
       the dataframe to be processed 
    
    Methods
    -------
    clean_df(df):
    Returns cleansed dataframe
    
    check_missing(df):
    Returns summary of missing values
    
    master_question():
    
    Returns dict for master question for future visualisation 
      
    """
    
    def __init__(self,df):
        self.df = df
    def clean_df(self):
        
        """
        Returns cleansed dataframe
            Parameters:
                    df(dataframe)
            Returns:
                    df(dataframe)
        """
        self.df.rename(columns=lambda x:re.sub('[^A-Za-z0-9]+',' ',x).strip().replace(' ','_').lower(),inplace=True)
        self.df.replace('\r\n','',regex=True,inplace=True)

        #convert the columns with % string value to numeric cardinal values
        percentage_columns=self.df.columns.values.tolist()[4:]
        self.df.loc[:,percentage_columns]=self.df.loc[:,percentage_columns].apply(lambda x:x.str.rstrip('%').astype(float)/100)

        #drop NA where col 'question' and most other cols are null
        self.df.drop(self.df[(self.df[self.df.columns[4]].isnull())&(self.df[self.df.columns[5]].isnull()
                                                                 &(self.df.question.isnull()))].index,inplace=True)
        self.df.drop('awareness_understanding_effectiveness_of_your_bu',axis=1,inplace=True)

        #Create order for col 'question' so that questions are in ascending order
        self.df['FirstOrder']=[d[0][0] for d in self.df.question_number.str.split()]
        self.df['SecondOrder']=[d[0][2] for d in self.df.question_number.str.split()]
        self.df.sort_values(['FirstOrder','SecondOrder'],inplace=True)
        self.df=self.df.reset_index(drop=True)
        
        #Cleanse the response distribution
        respsonse_mapping={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,
    
    'I do not have deep enough knowledge of this Priority to rate our implementation.':'LowAwareness',
'I understand this Priority but have not been at CGI long enough to have a perspective on the effectiveness of our implementation.':'ShortTenure',
        'I do not have a perspective on this topic.':'NoPerspective',
     'I do not have a perspective on this topic':'NoPerspective',
     'I do not have deep enough knowledge of this resource to rate its effectiveness':'LowAwareness',
     'I did not attend a Town Hall or team meeting in the past year':'NotAttended',
     'More than two years':'Over2years',
     'More than one year':'Over1year',
     'This topic is not relevant to my role':'NotRelevant',
     'Act as a trusted advisor to our clients by exercising independence in line with our values and code of ethics for our solution and partner recommendations as we build and manage an expansive technology ecosystem focused on creating sustainable business outcomes for our clients through global and specialized third-party relationships.':'TrustedAdvisor',
     'Attract, engage, proactively develop, and retain highly skilled business and IT experts in support of evolving market needs identified through our client relationships and in the Voice of Our Clients program, including to provide global industry capabilities.':'TalentManagement',
     'Make the Member Partnership Management Framework (MPMF) a reality at all levels, with increased emphasis on sustaining a culture that empowers members to make decisions, strengthens relationships and mutual trust, supports two-way communication, and engages members in all aspects of the company, such as business development, recruiting, delivery, and innovation.':'Empowerement',
     'In line with our Values, accelerate diversity, equity, and inclusion by taking relevant and timely action locally and globally to increase the sense of belonging for all members.':'Inclusion',
     'Provide a member-centric environment that promotes collaboration, flexibility and enhances member experience and productivity through integrated digital tools and physical and virtual workspaces. ':'Synergy',
     '''Promote members' health and wellbeing, including mental health and work-life balance, through global and local programs. ''':'Wellbeing',
     '''Constantly and consistently ensure visibility and recognition of CGI's local and global capabilities and expertise, thought leadership, and innovation internally and externally: a. with clients, by proactively and regularly bringing them innovative ideas, processes, methodologies, intellectual property, and tools to address their business goals and challenges through:  proximity-based and executive project relationships Executive Insights Exchange programs in each BU, including global delivery participation, focused on our end-to-end capabilities. b. within and across our targeted industry sectors, by showcasing innovative solutions and approaches to address industry and technology challenges through social media, industry and technology forums, our global alliance partners, and other market influencers. ''':'Visibility',
     'Consistent with the CGI client proximity metro market model, attract, develop and retain a sufficient number of members in the relevant industries for work requiring close interaction with clients, such as business and strategic IT consulting; project management; agile delivery methodology and enterprise and system architecture and design.':'Engagement',
     'Leverage our industry-vertical organization within and across metro markets and global industry councils to enable our members at all levels to grow their business domain expertise.':'Specialization',
     '''Create, acquire, evolve and deliver value added solutions, including IP and practices, targeted at supporting clients' 'business and operating activities in our selected verticals.''':'Innovation',
     '''Develop and expand our members' knowledge and expertise in technology, innovation and business-domain focused intellectual property solutions through participation in global IT communities of practice and by leveraging emerging technology and digitization practices and industry blueprints, and our global alliance partners certifications. ''':'Capability',
     '''Enable, invest in, and promote knowledge sharing and transfer between subject matter experts in industry verticals, business domains, and technology disciplines both locally and throughout CGI's global operations.''':'Collaboration',
     'Ensure full and consistent application of the Client Partnership Management Framework (CPMF) throughout our end-to-end services to: a. Proactively and consistently have meaningful client interactions to continuously build trust, including through high-quality Client Satisfaction Assessment Program (CSAP) meetings and engagement reviews; and b. Foster operational excellence, including by educating and coaching members so they can help reduce red engagements by quickly detecting, evaluating, and escalating issues to bring the right resources to bear to resolve issues; and c. Capturing learnings from green engagements and propagating across the organization.':'Excellence',
      '''Ensure all leaders and members work together to identify opportunities designed to help members build their skills, expertise, and careers in line with the evolving needs of clients and CGI and in line with our intrapreneurship value, including through: projects and assignments for on-the-job learning, including proactive rotations across clients, projects, and internal functions, delivery centers of excellence, and urban metro markets augmented by training, internships and mentorships; individual development plans to strengthen business domain and emerging technology expertise, and member competencies; participation in synergy sessions, project reviews, proposal development, business development meetings, industry engagements, etc., in order to foster engagement and broaden their career opportunities; providing opportunities to contribute to CGI's body of knowledge and continuous improvement by sharing their experience and expertise; and by sharing and discussing our strategic priorities and directions, including how all members can play a part in achieving them.''':'Development',
       'Proactively invest in identifying, building, and strengthening sustained relationships with business and IT executives, key decision makers, and influencers including procurement and analysts through a proactive, consultative approach. ':'Engagement',
       'Foster alignment and constantly seek the best equilibrium for all three stakeholders and our communities where we live and work by applying the principles, policies, and frameworks in the CGI Management Foundation and in line with our value of Financial Strength.':'Balance',
       '''Increase our members' knowledge and understanding of the content in the CGI Management Foundation through learning, mentoring and certification in CGI Academia. ''':'Learning',
       '''Maintain focus on continuous improvement of company performance by:\t\tMaking operating results visible at all levels;\t\t\tProviding business context to enhance understanding; and\t\t\tIdentifying, seeking input on, and taking rapid appropriate action to improve results.\t''':'Performance',
       'Maintain focus on continuous year over year improvement of cash management and financial performance, including improvement of our profitable growth measures. ':'Profitability',
       'Proactively identify and act urgently to prevent, optimize or remedy underperforming contracts, intellectual property, and other assets as well as optimize utilization of existing resources. ':'Optimization',
       '''Safeguard CGI, our reputation, and our three stakeholders' interests through the full global implementation of holistic security, data protection, privacy, and business continuity and sustainability practices in day-to-day operations at every level. ''':'Security',
       'We champion digital inclusion for all citizens, taking actions locally to improve access to technology and business education and mentoring in order to help everyone be successful in a digital society. ':'Inclusion',
       '''We commit to positively contribute to society by leveraging our members' personal engagement and IT and business expertise through investment in social impact projects, local economic growth initiatives, and by actively supporting local Business Unit pro bono engagements.''':'Social Impact',
       'Implement training programs for members at all levels to ensure and sustain business acumen, including business development, client relationship building, technical and industry skills, financial literacy and disciplined contract management, and delivery excellence.':'Proficiency',
       'Bid & Proposal preparation':'Proposal',
       'Empower members to live the Dream at all levels, in line with our Values, by fostering a culture of ownership, respect, trust, engagement, recognition, growth, purpose and celebration of success. ':'Empowerement',
       'Act in full transparency to our members by providing them on a regular basis with results as per Managing for Excellence metrics, including gaps to full Profit Participation Plan(PPP) distribution, explain context and discuss action plans and how they can contribute to meeting these objectives.':'Transparency',
       'Yes, based on a direct recommendation from my client':'Recommendation',
       '''Yes, based on my perspective of CGI's business or market trends''':'Perspective',
       'Yes, based on a recommendation from my client and my perspective':'Validation',
       'Orientation for new members':'Onboarding',
       'Preparing for a client meeting and/or presenting to clients':'Presentation',
       'Reference during 1:1 Member Satisfaction Assessment Program (MSAP) meetings':'Feedback',
       'Learning about our frameworks and processes':'Familiarization',
       'Training for team members':'Development',
       'Reference for managing operations on a day-to-day basis':'Guidance',
       'I do not believe this benchmarking is relevant for my client.':'Irrelevant',
       'Follow-up meeting to share insights with clients who participated':'Debriefing',
       'Executive Insights Exchange meetings':'Collaboration',
       'Thought leadership materials, including industry presentations':'Thought leadership',
       'Client Development Planning':'Strategy',
       'Annual Business Planning':'Strategy',
       'We demonstrate our commitment to an environmentally sustainable world through projects delivered in collaboration with clients, our solutions and through our operating and transportation practices, supply chain management and community service activities.':'Sustainability',
        '''Benchmarking initiatives, including for CGI's "Journey to World Class IT" offering''':'Benchmarking',
        '''Foster alignment and constantly seek the best equilibrium for all three stakeholders and our communities where we live and work by applying the principles, policies, and frameworks in the CGI Management Foundation and in line with our value of Financial Strength. ''':'Harmony',
        '''Create, acquire, evolve and deliver value added solutions, including IP and practices, targeted at supporting clients' business and operating activities in our selected verticals.''':'Innovation'                  
                          
                          }
        
        self.df['response']=self.df.response_distribution.map(respsonse_mapping)
        self.df.loc[self.df.response=='LowAwareness','response_distribution']='LowAwareness'
        self.df.loc[self.df.response=='ShortTenure','response_distribution']='ShortTenure'
        self.df.loc[self.df.response=='NoPerspective','response_distribution']='NoPerspective'
        self.df.loc[self.df.response=='NotAttended','response_distribution']='NotAttended'
        self.df.loc[self.df.response=='Over2years','response_distribution']='Over2years'
        self.df.loc[self.df.response=='Over1year','response_distribution']='Over1year'    
        self.df.loc[self.df.response=='NotRelevant','response_distribution']='NotRelevant'
        self.df.loc[self.df.response=='TrustedAdvisor','response_distribution']='TrustedAdvisor'
        self.df.loc[self.df.response=='TalentManagement','response_distribution']='TalentManagement'
        self.df.loc[self.df.response=='Empowerement','response_distribution']='Empowerement'
        self.df.loc[self.df.response=='Inclusion','response_distribution']='Inclusion'
        self.df.loc[self.df.response=='Synergy','response_distribution']='Synergy'
        self.df.loc[self.df.response=='Wellbeing','response_distribution']='Wellbeing'
        self.df.loc[self.df.response=='Visibility','response_distribution']='Visibility'
        self.df.loc[self.df.response=='Engagement','response_distribution']='Engagement'
        self.df.loc[self.df.response=='Specialization','response_distribution']='Specialization'
        self.df.loc[self.df.response=='Innovation','response_distribution']='Innovation'
        self.df.loc[self.df.response=='Capability','response_distribution']='Capability'
        self.df.loc[self.df.response=='Collaboration','response_distribution']='Collaboration'
        self.df.loc[self.df.response=='Excellence','response_distribution']='Excellence'
        self.df.loc[self.df.response=='Development','response_distribution']='Development'
        self.df.loc[self.df.response=='Engagement','response_distribution']='Engagement'
        self.df.loc[self.df.response=='Balance','response_distribution']='Balance'
        self.df.loc[self.df.response=='Learning','response_distribution']='Learning'
        self.df.loc[self.df.response=='Performance','response_distribution']='Performance'
        self.df.loc[self.df.response=='Profitability','response_distribution']='Profitability'
        self.df.loc[self.df.response=='Optimization','response_distribution']='Optimization'
        self.df.loc[self.df.response=='Security','response_distribution']='Security'
        self.df.loc[self.df.response=='Inclusion','response_distribution']='Inclusion'
        self.df.loc[self.df.response=='Social Impact','response_distribution']='Social Impact'
        self.df.loc[self.df.response=='Proficiency','response_distribution']='Proficiency'
        self.df.loc[self.df.response=='Proposal','response_distribution']='Proposal'
        self.df.loc[self.df.response=='Empowerement','response_distribution']='Empowerement'
        self.df.loc[self.df.response=='Transparency','response_distribution']='Transparency'
        self.df.loc[self.df.response=='Recommendation','response_distribution']='Recommendation'
        self.df.loc[self.df.response=='Perspective','response_distribution']='Perspective'
        self.df.loc[self.df.response=='Validation','response_distribution']='Validation'
        self.df.loc[self.df.response=='Onboarding','response_distribution']='Onboarding'
        self.df.loc[self.df.response=='Presentation','response_distribution']='Presentation'
        self.df.loc[self.df.response=='Feedback','response_distribution']='Feedback'
        self.df.loc[self.df.response=='Familiarization','response_distribution']='Familiarization'
        self.df.loc[self.df.response=='Development','response_distribution']='Development'
        self.df.loc[self.df.response=='Guidance','response_distribution']='Guidance'
        self.df.loc[self.df.response=='Irrelevant','response_distribution']='Irrelevant'
        self.df.loc[self.df.response=='Debriefing','response_distribution']='Debriefing'
        self.df.loc[self.df.response=='Collaboration','response_distribution']='Collaboration'
        self.df.loc[self.df.response=='Benchmarking','response_distribution']='Benchmarking'
        self.df.loc[self.df.response=='Thought leadership','response_distribution']='Thought leadership'
        self.df.loc[self.df.response=='Strategy','response_distribution']='Strategy'
        self.df.loc[self.df.response=='Sustainability','response_distribution']='Sustainability'
        self.df.loc[self.df.response=='Harmony','response_distribution']='Harmony'
        self.df.drop('response',axis=1,inplace=True)
        
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
            percent=(self.df.isnull().sum()/self.df.isnull().count()).sort_values(ascending=False)
            missing=pd.concat([total,percent],axis=1,keys=['total','percent'])

            data_type=[]
            for col in self.df.columns:
                dtype=str(self.df[col].dtype)
                data_type.append(dtype)
            missing['Type']=data_type
            return missing
        else:
            return(False) 
        
    def master_question(self):
        #drop rows that are in master question rows and convert master question rows to dict for later visualisation 
        master_question=self.df.loc[(self.df.response_distribution==' '),['question_number','question']].reset_index(drop=True)
        master_question_dict=dict(zip(self.df.question_number,self.df.question))
        self.df.drop(self.df[self.df.response_distribution==' '].index,inplace=True)    
        self.df=self.df.reset_index(drop=True)
        return master_question_dict
    

def convert_to_list(counter):
    """
    Returns unique list of counter
       Parameters: counter(int)
       Returns: sorted_l (list)
    """
    tmp=set(counter)
    sorted_l=list(tmp)
    return sorted(sorted_l)

def convert_percentage(df):
    """
    Returns converted number as percetange for y axis visualisation 
       Parameters: df (dataframe)
       Returns: df (dataframe)
    """
    for c in df.columns[3:-2]:
        df.loc[:,c]=(df.loc[:,c]*100).astype(float)
    return df

def plot_barcharts(df): 
    """
    Returns barplot to visualise each job/sector reponse distribution 
       Parameters: df(dataframe)
       Returns: plot
    """
    plt.figure(figsize=(10,6))
    if len(df.columns)>9:
        df=df.copy()[['question_number', 'question', 'response_distribution', 'cgi','uka_south_and_midlands','central_and_south_metros_l8','hr_solutions_l8','shell_lvc_l8',
                               'FirstOrder', 'SecondOrder' ]]
    
    y_columns=df.columns[4:-2].values.tolist()
    g=df.plot(x='response_distribution',y=y_columns,kind='bar',fontsize=10)
    plt.xticks(rotation=60)
    plt.legend(fontsize=10)
    g.set_title(df['question_number'].values[0],fontsize=14)
    g.set_xlabel('response distribution',fontsize=14)
    g.set_ylabel('percentage(%)',fontsize=14)
    
    return plt.show()

def processing_visualisation(df,counter):
    """
    Returns processed and cleansed df to identify each question as a single df for later visualisation
       Parameters: df (dataframe)
                   counter(int): the index to identify the start and end index for each question so each question can be grouped as a                          individual dataframe
       Returns: master_df (dataframe)
                   
    """
    sorted_counter=convert_to_list(counter)
    df_tmp={}
    for i in range(len(sorted_counter)):
        if i==0:
            df_tmp[i]=df[0:sorted_counter[i]]
        else:
            df_tmp[i]=df[sorted_counter[i-1]:sorted_counter[i]]
    
    master_df=[]
    for k, v in df_tmp.items():
        df=pd.DataFrame(df_tmp.items())[1][k]
        master_df.append(df)
    for i in range(len(master_df)):
        master_df[i]=convert_percentage(master_df[i])
    return master_df

def clean_summary(df):
    df.rename(columns=lambda x:re.sub('[^A-Za-z0-9]+',' ',x).strip().replace(' ','_').lower(),inplace=True)
    numerical_columns=['2023_respondent','2022_respondent']
    for c in numerical_columns:
        df[c]=df[c].apply(lambda x:re.sub('\D','',x)).astype(int)
    return df 

def find_counters(df, column_name):
    df=df.reset_index(drop=True)
    counter_list = []
    i = 0
    while i < len(df):
        counter = i
        new_initial = df[column_name][counter]
        while i < len(df) and df[column_name][i] == new_initial:
            i += 1
            continue
        else:
            counter = i
            print(counter)
            counter_list.append(counter)
    return counter_list

def calculate_avg(df,column_len):
    calculation_df=pickle.loads(pickle.dumps(df))
    cal_df=[]
    q_df=[]
    
    for i in range(len(calculation_df)):
        if '10' in calculation_df[i]['response_distribution'].values:
            calculation_df[i].loc[~calculation_df[i].response_distribution.isin(list(map(str,list(range(1,11))))),'response_distribution']=0
            calculation_df[i]['response_distribution']=calculation_df[i]['response_distribution'].map(int)
            for c in calculation_df[i].columns[4:-2]:
                calculation_df[i]['calculated_'+c]=calculation_df[i][c]*calculation_df[i]['response_distribution']
            avg_s=round((calculation_df[i].iloc[:,-column_len:].sum()/100).sum()/3,2)
            q_df.append(calculation_df[i]['question_number'].values[0])
            cal_df.append(avg_s)
    tmp=pd.DataFrame(pd.Series(dict(zip(q_df,cal_df)))).reset_index().rename(columns={'index':'Q',0:'Avg_Score'})
    return tmp