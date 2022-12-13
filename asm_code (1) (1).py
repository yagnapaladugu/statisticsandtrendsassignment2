#!/usr/bin/env python
# coding: utf-8

# #importing library 

# *******************Matplotlib is a Python package that allows you to make static, animated, and interactive visualisations. 
# Matplotlib makes both easy and difficult tasks possible. Create plots that are suitable for publication.
#  Create interactive graphs that can be zoomed, paned, and updated.****
# 
#  ***NumPy can be used to conduct a wide range of array-based mathematical calculations. 
# 
# 
#  ***Pandas is the most extensively used Python data analysis package. It provides highly optimised performance using back-end source code.
# 

# In[21]:


import pandas  as pd
#importing pandas library
import matplotlib.pyplot as mtp_plt 
# importing matplot library 
import numpy as np
# importing numpy library 


# # creating function 

# In[22]:


# create function 
def read_data(Data_file):  
    dataframe1=pd.read_csv(Data_file,skiprows=4)
     # read dataset
    dataframe2=dataframe1.drop(['Country Code', 'Unnamed: 66', 'Indicator Code'],axis=1)  
    # droping unuse columns 
    dataframe3=dataframe2.set_index("Country Name") 
    #  set index name
    dataframe4=dataframe3.T    
    # tranpose dataset                 
    dataframe4.reset_index(inplace=True) 
    # reset index name 
    dataframe4.rename(columns = {'index':'Year'}, inplace = True)
    # set index name  
    return dataframe1,dataframe4 
     # return data frame


# In[23]:


Data_file="/content/API_19_DS2_en_csv_v2_4700503.csv" 
# read dataset path

data_frm1,data_frm2=read_data(Data_file) 
# read two dataset frame


# In[24]:


data_frm1.head()
# printing data frame 1 


# In[25]:


data_frm2.head() 
# printing data frame 2 


# #Population growth (annual %) correlation with country and year 

# In[26]:


data_corr= data_frm1[data_frm1['Indicator Name']=='Population growth (annual %)']
data_corr1 = data_corr.pivot_table(index=['Country Name'], values = ['1985', '1995', '2015'])
data_corr1.head(10)


# #PLOTTING BAR GRAPH 

# #Graph for Electricity production from oil sources (% of total)

# In[27]:


# we are plotting bar graph 
Bar_Dataset = data_frm1[data_frm1['Indicator Name']=='Electricity production from oil sources (% of total)']  
Bar_Dataset_1 = Bar_Dataset.pivot_table(index=['Country Name'], values = ['1970', '1980', '1990', '2010', '2020']) 
# define color, figure size fpor graph.
mtp_plt.rcParams.update({'font.size': 18}) 
Bar_Dataset_1.head(15).plot.bar(color=['Aqua','red','Blue','Chartreuse','orange'],figsize=(20,10))  
# set x_lable for graph
mtp_plt.xlabel('Name Of The Nation') 
# set y_label for graph.
mtp_plt.ylabel('Comparisons')
# set title  name 
mtp_plt.title('Electricity production from oil sources (% of total)')  
mtp_plt.show(); 


# #Graph for CO2 emissions from liquid fuel consumption (% of total)

# In[28]:


# we are plotting bar graph 
dataset_2 = data_frm1[data_frm1['Indicator Name']== 'CO2 emissions from liquid fuel consumption (% of total)']  
# Using pivot table set index and values for making the graph.
dataset_bar_2 = dataset_2.pivot_table(index=['Country Name'], values = ['1970', '1980', '1990', '2010', '2020'])  
# define color and figure size for graph.
dataset_bar_2.head(15).plot.bar(color=['Blue','GreenYellow','DarkOrange','BlueViolet','Fuchsia'],figsize=(20,10))  

mtp_plt.xlabel('Name Of The Nation') 
mtp_plt.ylabel('Comparisions') 
# set title  name 
mtp_plt.title('CO2 emissions from liquid fuel consumption (% of total)')  
mtp_plt.show(); 


# #Time series Analysis

# #plotting line figure  for CO2 emissions (metric tons per capita)

# In[29]:



Dataframe = data_frm1[data_frm1['Indicator Name']=='CO2 emissions (metric tons per capita)']  
# index set.
Dataframe = Dataframe.set_index("Country Name") 
# dropping all the columns.
Dataframe = Dataframe.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'],axis=1)
Dataframe = Dataframe.T
# reset index.
Dataframe = Dataframe.reset_index() 


# In[30]:


# every single variable used in the tabular
Dataframe_10=Dataframe.pivot_table(index=['index'], values=['Russian Federation', 'Rwanda', 'South Asia', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Serbia'])  
# set figure size 
mtp_plt.figure(figsize = (18,8)) 
mtp_plt.plot(Dataframe_10.head(20),'-..')
# set rotation of xticks. 
mtp_plt.xticks(rotation=45) 
# set all the legends.
mtp_plt.legend(['Russian Federation', 'Rwanda', 'South Asia', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Serbia'],bbox_to_anchor = (1.0,1.1),ncol=1) 
mtp_plt.xlabel('Year')
# set x axis year 
mtp_plt.ylabel('Comparison ')
# set y axis  price 
mtp_plt.title('CO2 emissions (metric tons per capita)') 
# set title name 
mtp_plt.show() # showing graph 


# #plotting line figure for Electricity production from oil sources (% of total)

# In[31]:


# gathering all relevant facts on  Electricity production from oil sources (% of total)
DATA_feame_10 = data_frm1[data_frm1['Indicator Name']== 'Electricity production from oil sources (% of total)']  
# we set index as country name.
DATA_feame_11 = DATA_feame_10.set_index("Country Name") 
# dropping all the unnecessary columns
DATA_feame_12=DATA_feame_11.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'],axis=1)
DATA_feame_13=DATA_feame_12.T
# reseting the index.
DATA_feame_14=DATA_feame_13.reset_index() 


# In[32]:


# We are selecting all data using pivot table like index and values.
dataframe_25=DATA_feame_14.pivot_table(index=['index'], values=[ 'Russian Federation', 'Rwanda', 'South Asia', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Serbia'])  
# set figure size
mtp_plt.figure(figsize = (18,8))  
# print top 20 data.
mtp_plt.plot(dataframe_25.head(20),'-..' ) 
# set rotation for xticks.
mtp_plt.xticks(rotation=45) 
# set lagend.
mtp_plt.legend([ 'Russian Federation', 'Rwanda', 'South Asia', 'Saudi Arabia', 'Sudan', 'Senegal', 'Singapore', 'Solomon Islands', 'Sierra Leone', 'El Salvador', 'San Marino', 'Somalia', 'Serbia'],bbox_to_anchor = (1.0,1.1),ncol=1) 
mtp_plt.xlabel('Year')
# set x axis  year 
mtp_plt.ylabel('Comparisons')
# set y axis  price 
mtp_plt.title( 'Electricity production from oil sources (% of total)') 
# set title name 
mtp_plt.show() 
# showing graph 


# #CORRELATION MATRIX 

# In[33]:


#creating function for country
def function(Country):
  data_corr_mat= data_frm1[data_frm1['Country Name']==f'{Country}'] 
  data_corr_mat = data_corr_mat.drop(['Country Code', 'Indicator Code', 'Country Name', 'Unnamed: 66'],axis=1) 
  data_corr_mat = data_corr_mat.T
  data_corr_mat1=data_corr_mat.iloc[0] 
  data_corr_mat=data_corr_mat[1:] 
  data_corr_mat.columns=data_corr_mat1
  data_corr_mat = data_corr_mat.reset_index(drop=True)
  return data_corr_mat


# In[34]:


df_country1=function('Saudi Arabia') 
# fatch country name 
df_country1.to_csv('Saudi Arabia.csv') 
# create new data set file 
df_country2=pd.read_csv('/content/Saudi Arabia.csv') 
# read dataset for Saudi Arabia country 
df_country3=df_country2.drop(['Unnamed: 0'],axis=1) 
# drop columns 
data_fill = df_country3.fillna(0) 
# fill null values 


# In[35]:


#define labels  
df_matrix1 = data_fill[['CO2 emissions from liquid fuel consumption (% of total)', 'CO2 emissions from liquid fuel consumption (kt)', 'CO2 emissions (kt)', 'CO2 emissions (kg per 2015 US$ of GDP)', 'CO2 emissions from gaseous fuel consumption (% of total)', 'CO2 emissions from gaseous fuel consumption (kt)', 'CO2 intensity (kg per kg of oil equivalent energy use)','Urban population', 'Urban population growth (annual %)', 'Population, total', 'Population growth (annual %)']]  


# In[36]:



df_corr1=df_matrix1.corr() #correlation
df_corr1.head() 


# #Plot figure for Saudi Arabia

# In[37]:


# plotting corr matrix 
arr=df_corr1.to_numpy()
labs=df_matrix1.columns
fig, Axis = mtp_plt.subplots(figsize=(20,12))
 # set figure size 
im = Axis.imshow(df_corr1,cmap="PuBu")
# labels lenth 
Axis.set_xticks(np.arange(len(labs)))
Axis.set_yticks(np.arange(len(labs)))

Axis.set_xticklabels(labs)
Axis.set_yticklabels(labs)

mtp_plt.setp(Axis.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor") 

for i in range(len(labs)):
    for j in range(len(labs)):
        text = Axis.text(j, i, round(arr[i, j],2), ha="center", va="center", color="black") 
# set title name saudi arabia
Axis.set_title("Saudi Arabia") 
# Showing matrix 
mtp_plt.show() 


# #Plot figure for Singapore

# In[38]:


df_country4=function('Singapore')
 # fatch country name 
df_country4.to_csv('Singapore.csv') 
# create new data set file 
df_country5=pd.read_csv('/content/Singapore.csv') 
# read dataset for Singapore country 
df_country6=df_country5.drop(['Unnamed: 0'],axis=1) 
# drop unnamed columns 
data_fill= df_country6.fillna(0) 
# fill null values 


# In[39]:


# define labels 
df_matrix2= data_fill[['CO2 emissions from liquid fuel consumption (% of total)', 'CO2 emissions from liquid fuel consumption (kt)', 'CO2 emissions (kt)', 'CO2 emissions (kg per 2015 US$ of GDP)', 'CO2 emissions from gaseous fuel consumption (% of total)', 'CO2 emissions from gaseous fuel consumption (kt)', 'CO2 intensity (kg per kg of oil equivalent energy use)','Urban population', 'Urban population growth (annual %)', 'Population, total', 'Population growth (annual %)']] 


# In[40]:


# showing correlation  data
df_corr2=df_matrix2.corr() 
df_corr2.head() 


# In[41]:


# plotting correlation matrix 
arr=df_corr2.to_numpy()
labs=df_matrix2.columns
fig, Axis = mtp_plt.subplots(figsize=(20,12)) 
# set figure size 
im = Axis.imshow(df_corr2,cmap="Greys_r")
# lables lenth 
Axis.set_xticks(np.arange(len(labs)))
Axis.set_yticks(np.arange(len(labs)))

Axis.set_xticklabels(labs)
Axis.set_yticklabels(labs)

mtp_plt.setp(Axis.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor") 

for i in range(len(labs)):
    for j in range(len(labs)):
        text = Axis.text(j, i, round(arr[i, j],2), ha="center", va="center", color="black") 
# set title name  Singapore
Axis.set_title("Singapore") 
mtp_plt.show()  
# showing graph 


# In[41]:




