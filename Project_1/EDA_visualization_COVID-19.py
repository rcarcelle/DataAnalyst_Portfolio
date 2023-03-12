#!/usr/bin/env python
# coding: utf-8

# ## COVID-19 Data Repository
# ## EDA + Visualization

# #### The project aims to clean and analyze a COVID-19 dataset from Johns Hopkins University, 
# #### which includes data from multiple countries and regions. The first step will be to clean 
# #### and preprocess the data to ensure its accuracy and consistency. Then, various visualizations 
# #### and statistical analyses will be conducted to gain insights into the trends and patterns of 
# #### COVID-19 cases, deaths, and recoveries. The project will also explore the impact of various factors, 
# #### such as demographics and healthcare systems, on the spread and severity of the disease. 
# #### Ultimately, the project aims to provide valuable information and insights to aid in the global fight against COVID-19.

# #### In this exercise we will analyze 
# #### the Johns Hopkins CoronaVirus dataset. (https://github.com/CSSEGISandData/COVID-19)
# 
# Path of the 2 files to download:
# - https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
# 
# - https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv

# In[30]:


### import libraries

'''
install:
- plotly
- calmap
- folium
- wget
'''

#analysis
import pandas as pd 
import numpy as np

#visualization
import calmap
import folium

#palettes
confirmados='#393e46'
fallecidos='#ff2e63'

import wget
import os
import plotly.express as px


# In[31]:


urls = ['https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', 
        'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv']
for url in urls:
    filename = wget.download(url, os.getcwd() )


# In[32]:


cf_df=pd.read_csv("time_series_covid19_confirmed_global.csv")
de_df=pd.read_csv("time_series_covid19_deaths_global.csv")


# In[33]:


de_df.info()
de_df.head()


# In[34]:


cf_df.info()
cf_df.head()


# In[35]:


cf_df.shape


# In[36]:


de_df.shape


# In[37]:


cf_df.columns
cf_df.columns[:4]


# In[38]:


# 2. Let's transpose the table in such a way that I get the dates as rows:

# actual format:
# country city date1 date2 date3...

# recomended format:
# date1 country city
# date2 country city
# date3 country city


# In[39]:


columnas_fechas=dates=cf_df.columns[4:]

cf_df_trans= cf_df.melt(id_vars=['Province/State','Country/Region','Lat','Long'],
                         value_vars=columnas_fechas, var_name='Date', value_name='Confirmed')

cf_df_trans

de_df_trans= de_df.melt(id_vars=['Province/State','Country/Region','Lat','Long'],
                         value_vars=columnas_fechas, var_name='Date', value_name='Deaths')

de_df_trans


# In[40]:


cf_df_trans.head()
de_df_trans.head()

data= pd.merge(cf_df_trans,de_df_trans, how="inner",on=["Province/State", "Country/Region", "Date", "Lat", "Long" ])


# In[41]:


data.shape


# In[42]:


#Sort the columns to be: date country province latitude longitude confirmed deaths
data=data.reindex(["Date","Country/Region","Province/State","Lat","Long","Confirmed","Deaths"], axis=1)


# In[43]:


data


# In[44]:


data.info()


# In[45]:


#Convert the date to datetime
data["Date"]=data["Date"].apply(pd.to_datetime)


# In[46]:


data.info()


# In[47]:


#Check nulls
data.isnull().sum()


# In[48]:


#See provinces that have information
data["Province/State"].value_counts()


# In[49]:


data["Province/State"].value_counts()


# In[50]:


data[ data["Province/State"].isnull()==False ]


# In[51]:


#Check if the country is separated from the provinces (grouped)
data[ (data["Country/Region"]=="United Kingdom") & (data["Province/State"].isnull()) ]


# In[52]:


#See the recovered ones from which countries are the null (or may be provinces!)
data.isnull().sum()


# In[53]:


#Let's rename the columns
data.rename(columns={"Country/Region":"Pais"}, inplace=True)
data.rename(columns={"Province/State":"Provincia"}, inplace=True)
data.rename(columns={"Date":"Fecha"}, inplace=True)


# In[54]:


data.head()
Lista_Paises=data["Pais"].unique().tolist()


# In[55]:


Lista_Paises


# In[56]:


print(Lista_Paises)
print("Son un total de %s paises afectados por el COVID" %len(Lista_Paises))


# In[57]:


data["Active"]=data["Confirmed"]-data["Deaths"]


# In[58]:


#Create an array of the cases
casos=["Confirmed","Deaths","Active"]


# In[59]:


#Let's fill in the nulls
data["Provincia"].fillna("Península",inplace=True)


# In[60]:


for i in casos:
    data[i].fillna(0,inplace=True)


# In[61]:


data[data["Lat"].isnull()]["Pais"].value_counts()


# In[62]:


data[data["Long"].isnull()]["Pais"].value_counts()


# In[63]:


data["Lat"]=np.where(data["Pais"]=="China",35.86166, data["Lat"])


# In[64]:


data["Long"]=np.where(data["Pais"]=="China",104.195397, data["Long"])


# In[65]:


data["Lat"]=np.where(data["Pais"]=="Canada", 56.130366 , data["Lat"])


# In[66]:


data["Long"]=np.where(data["Pais"]=="Canada", -106.346771 , data["Long"])


# In[67]:


data.isnull().sum()


# In[68]:


data[data["Pais"]=="China"]


# #### Let's create two datasets:
# - data_ultima_fecha= will contain only the rows of the last day, it is by country and province. 
# - data_ultima_fecha_pais= will contain only the rows of the last day, it will be by country only

# In[69]:


data_ultima_fecha=data[data["Fecha"]==max(data["Fecha"])].reset_index()
data_ultima_fecha


# In[70]:


data_ultima_fecha_pais=data_ultima_fecha.groupby("Pais")[["Confirmed","Deaths","Active"]].sum().reset_index()
data_ultima_fecha_pais


# In[71]:


data_ultima_fecha_pais[ (data_ultima_fecha_pais["Pais"]=="Spain") ]


# ********************************************************************
#                                 MAPS
# ********************************************************************

# In[72]:



mapa_global1 = folium.Map(location=[39.00, -3.703790], zoom_start=5)

for lat, lon, value, name in zip(data_ultima_fecha['Lat'], data_ultima_fecha['Long'], data_ultima_fecha['Confirmed'] ,data_ultima_fecha['Pais']):
        folium.CircleMarker([lat, lon],
                        radius=5,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),
                        color='red',
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(mapa_global1)

mapa_global1


# In[73]:


'''
Map with red circles of confirmed cases,
in this case it is enough to hover the mouse to display the information.
'''
mapa_global2 = folium.Map(location=[0, 0], tiles='cartodbpositron',
               min_zoom=1, max_zoom=4, zoom_start=1)

for i in range(0, len(data_ultima_fecha)):
    folium.Circle(
        location=[data_ultima_fecha.iloc[i]['Lat'], data_ultima_fecha.iloc[i]['Long']],
        color='crimson', 
        tooltip =   '<li><bold>Country : '+str(data_ultima_fecha.iloc[i]['Pais'])+
                    '<li><bold>Province : '+str(data_ultima_fecha.iloc[i]['Provincia'])+
                    '<li><bold>Confirmed : '+str(data_ultima_fecha.iloc[i]['Confirmed'])+
                    '<li><bold>Deaths : '+str(data_ultima_fecha.iloc[i]['Deaths'])
                    ,
        radius=int(data_ultima_fecha.iloc[i]['Confirmed'])**0.75).add_to(mapa_global2)

mapa_global2


# In[77]:


mapa_global3 = px.choropleth(data_ultima_fecha_pais, locations="Pais", 
                    locationmode='country names', color="Confirmed", 
                    hover_name="Pais", range_color=[1,1e7],  
                    color_continuous_scale=["Green","Yellow","Red"], 
                    title='Countries with confirmed cases')


mapa_global3.show()


# In[ ]:




