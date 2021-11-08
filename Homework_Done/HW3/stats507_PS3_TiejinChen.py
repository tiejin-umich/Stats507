# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Homework 3
#
# ### Problem 0
# #### Data files
# link for both datasets is:
# 1. 2015:https://www.eia.gov/consumption/residential/data/2015/index.php?view=microdata
# and here is the superlink [2015](https://www.eia.gov/consumption/residential/data/2015/index.php?view=microdata)
# 2. 2009: https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata
# and here is the superlink [2009](https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata)
#
# link for the dataset download:
# 1. 2015: https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv
#
# 2. 2009:
# - data: https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv
# - Replicate weights:https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv
#
# #### Variables
# We will only use the following variables:
# - DOEID:Unique identifier for each respondent
# - REGIONC:Census Region
# - HDD65:Heating degree days in 2009(2015), base temperature 65F
# - CDD65:Cooling degree days in 2009, base temperature 65F
# - NWEIGHT: Final sample weight
# - BRRWT_i: i from 1 to 96, i-th replicate weights for 2015 data, not in 2009
# - brr_weight_i: i from 1 to 244, i-th replicate weights for 2009 data, not in 2015
#
#
# #### Weights and Replicate Weights
# Links to how to use replicate weights:
# 1. 2015: https://www.eia.gov/consumption/residential/data/2015/pdf/microdata_v3.pdf
# 2. 2009: https://www.eia.gov/consumption/residential/methodology/2009/pdf/using-microdata-022613.pdf
#
# From the documents, we know how to estimate variance:
# > $\hat{V}(\tilde{\theta}) = \frac{1}{R(1-\varepsilon)^2}\sum_{r=1}^{R}(\widehat{\theta_r}-\hat{\theta})^2$
#
# Hence, the estimation of standard error is:
#
# $se(\tilde{\theta}) =\sqrt{\hat{V}(\tilde{\theta})}=
# \sqrt{\frac{1}{R(1-\varepsilon)^2}\sum_{r=1}^{R}(\widehat{\theta_r}-\hat{\theta})^2}$
#
# Where $\theta$ is parameter of interest, $R$ is the number of replicate subsample,
# in 2009 data $R$ is 244 and in 2015 data $R$ is 96. $\varepsilon$ is Fay coefficient and
# we set this parameter to 0.5 following the instruction from document. $\widehat{\theta_r}$ is
# the estimation from r-th replicate subsample with r-th replicate weights. And $\hat{\theta}$ is
# the  estimation from full sample which means we will use NWEIGHT(Final sample weight) to estimate it.

# ### Problem 1
# #### Part a
# First we import all the packages we need to use

import pandas as pd
from os.path import exists
from scipy.stats import norm
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

# Then we put all the names of data files in a list ```data_filt```,
# put all the download links of data files in a list ```data_file_link```

datafile_09 = 'recs2009_public.csv'
datafile_09_link = 'https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv'
datafile_15 = 'recs2015_public_v4.csv'
datafile_15_link = 'https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv'
weight_09 = 'recs2009_public_repweights.csv'
weight_09_link = 'https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv'
data_file = [datafile_09,datafile_15,weight_09]
data_file_link = [datafile_09_link,datafile_15_link,weight_09_link]

# The following codes is to read the data into dataframe. If in the path this .py file
# we can not find the file, we will download it. And we save all the dataframe in a
# list ```store_data```, where ```store_data[0]``` is 2009 data, ```store_data[1]``` is
# 2015 data and ```store_data[2]``` is the replicate weights of 2009 data.

store_data = []
for i in range(3):
    if exists(data_file[i]):
        store_data.append(pd.read_csv(data_file[i],low_memory=False))
    else:
        temp = pd.read_csv(data_file_link[i],low_memory=False)
        temp.to_csv(data_file[i])
        store_data.append(temp)

# Then we take all the columns we need. And creating categorical types to ```REGIONC```.
# Here we think ```NWEIGHT``` is not replicate weights, hence the dataframe has this column.

col_name_both = ['DOEID','REGIONC','HDD65','CDD65','NWEIGHT']
cate_dict = {1:"Northeast", 2:"Midwest",3:"South",4:"West"}
data_09 = store_data[0].loc[:,col_name_both]
data_15 = store_data[1].loc[:,col_name_both]
data_09.loc[:,'REGIONC'] = pd.Categorical(data_09['REGIONC'].replace(cate_dict))
data_15.loc[:,'REGIONC'] = pd.Categorical(data_15['REGIONC'].replace(cate_dict))


# Finally, we get two dataframe ```data_09 ``` and ```data_15```
# which are our datasets for 2009 and 2015

data_09

data_15

# #### Part b
# In this part, we create two datasets with only ID and replicate weights.
# Here we stress that we don't think ```NWEIGHT```  as replicate weights.
# Hence the datasets in this part will not contain this columns.

rep_weight_09 = store_data[2].drop(['NWEIGHT','Unnamed: 0'],axis=1)
weight_name_15 = ['DOEID']
for i in range(1,97):
    weight_name_15.append('BRRWT{}'.format(i))
rep_weight_15 = store_data[1][weight_name_15]
rep_weight_09 = pd.melt(rep_weight_09,id_vars='DOEID')
rep_weight_15 = pd.melt(rep_weight_15,id_vars='DOEID')

# We get the result ```rep_weight_09``` and ```rep_weight_15``` with long format

rep_weight_09

rep_weight_15

# ### Problem 2
# #### Part a
# In this part, we will get the point estimation and 95% CI of the average number of heating
# and cooling degree days for residences in each Census region for both 2009 and 2015.
# First, we set ```DOEID``` as index for all the datasets we get in last problem.

data_09.set_index(['DOEID'], inplace=True,drop=True)
data_15.set_index(['DOEID'], inplace=True,drop=True)
rep_weight_09.set_index(['DOEID'], inplace=True,drop=True)
rep_weight_15.set_index(['DOEID'], inplace=True,drop=True)


# Then we use the following code with groupby-apply-combine method to get the point estimation 

def heat_data_point_esti(group):
    return (group['HDD65'] * group['NWEIGHT']).sum()/group['NWEIGHT'].sum()
def cool_data_point_esti(group):
    return (group['CDD65'] * group['NWEIGHT']).sum()/group['NWEIGHT'].sum()
heat_point_estimation_09 = data_09.groupby('REGIONC').apply(heat_data_point_esti)
cool_point_estimation_09 = data_09.groupby('REGIONC').apply(cool_data_point_esti)
heat_point_estimation_15 = data_15.groupby('REGIONC').apply(heat_data_point_esti)
cool_point_estimation_15 = data_15.groupby('REGIONC').apply(cool_data_point_esti)


# We use the same logic to calculate the standard error. However, the function we
# use in .apply also need .apply in itself to traverse two groupby result.
# So it is a little bit complex.

def weight_mean_two(group,col_name,other):
    def weight_mean(group_1,other_col):
        weight = group_1['value']
        #one = other_col/other_col
        one = pd.Series(1,index=other_col.index)
        return (weight*other_col).sum()/(weight * one).sum()
    value = group[col_name]
    return other.groupby('variable').apply(weight_mean,value)
heat_rep_var_temp_09 = data_09.groupby('REGIONC').apply(weight_mean_two,'HDD65',rep_weight_09)\
    .sub(heat_point_estimation_09,axis=0)
heat_rep_sd_09 = ((heat_rep_var_temp_09**2).sum(axis=1)/(244*0.25))**(0.5)
cool_rep_var_temp_09 = data_09.groupby('REGIONC').apply(weight_mean_two,'CDD65',rep_weight_09)\
    .sub(cool_point_estimation_09,axis=0)
cool_rep_sd_09 = ((cool_rep_var_temp_09**2).sum(axis=1)/(244*0.25))**(0.5)
heat_rep_var_temp_15 = data_15.groupby('REGIONC').apply(weight_mean_two,'HDD65',rep_weight_15)\
    .sub(heat_point_estimation_15,axis=0)
heat_rep_sd_15 = ((heat_rep_var_temp_15**2).sum(axis=1)/(96*0.25))**(0.5)
cool_rep_var_temp_15 = data_15.groupby('REGIONC').apply(weight_mean_two,'CDD65',rep_weight_15)\
    .sub(cool_point_estimation_15,axis=0)
cool_rep_sd_15 = ((cool_rep_var_temp_15**2).sum(axis=1)/(96*0.25))**(0.5)

# We get the result is ```heat_rep_sd_09```,```cool_rep_sd_09```,```heat_rep_sd_15```,
# ```cool_rep_sd_15```.And  ```heat_rep_sd_09``` means standard error of heating degree days
# in 2009 data, ```cool_rep_sd_15``` means standard error of cooling degree days in 2015 data.
#
# Then we can calculate 95% CI.

heat_point_estimation_09_lwr = heat_point_estimation_09-norm.ppf(0.975)*heat_rep_sd_09
heat_point_estimation_09_upr = heat_point_estimation_09+norm.ppf(0.975)*heat_rep_sd_09
cool_point_estimation_09_lwr = cool_point_estimation_09-norm.ppf(0.975)*cool_rep_sd_09
cool_point_estimation_09_upr = cool_point_estimation_09+norm.ppf(0.975)*cool_rep_sd_09
store_result_09 = [heat_point_estimation_09_lwr,heat_point_estimation_09,
                   heat_point_estimation_09_upr,cool_point_estimation_09_lwr,
                   cool_point_estimation_09,cool_point_estimation_09_upr]

heat_point_estimation_15_lwr = heat_point_estimation_15-norm.ppf(0.975)*heat_rep_sd_15
heat_point_estimation_15_upr = heat_point_estimation_15+norm.ppf(0.975)*heat_rep_sd_15
cool_point_estimation_15_lwr = cool_point_estimation_15-norm.ppf(0.975)*cool_rep_sd_15
cool_point_estimation_15_upr = cool_point_estimation_15+norm.ppf(0.975)*cool_rep_sd_15
store_result_15 = [heat_point_estimation_15_lwr,heat_point_estimation_15,
                   heat_point_estimation_15_upr,cool_point_estimation_15_lwr,
                   cool_point_estimation_15,cool_point_estimation_15_upr]


# We get the two list ```store_result_09``` and ```store_result_15```
# to store all the result, including point estimation,
# their lower bound of estimation and upper bound of estimation.
#
# Finally, We define a function ```create_table ``` to get a good table to show.

def create_table(store_result):
    estimation_heat = pd.concat(store_result[:3],axis=1)
    estimation_heat.columns =['lower bound of estimation', 'point estimation',
                              'upper bound of estimation']
    estimation_heat['Temperature Type'] = 'heat'
    estimation_cool = pd.concat(store_result[3:],axis=1)
    estimation_cool.columns =['lower bound of estimation', 'point estimation',
                              'upper bound of estimation']
    estimation_cool['Temperature Type'] = 'cool'
    estimation = pd.concat([estimation_heat,estimation_cool])
    estimation.index.name = 'Census Region'
    estimation.set_index('Temperature Type',append=True,inplace=True)
    return estimation.swaplevel()
table_09 = create_table(store_result_09)
table_15 = create_table(store_result_15)

# We get the result ```table_09``` and ```table_15```. Now we display them.

cap = """
<b> Table 1.</b> <em> Point estimation and 95%CI of 2009 data.</em>
When Temperature Type is 'heat', then the data is the estimation 
and 95%CI of heating degree days.
When Temperature Type is 'cool', then the data is the estimation 
and 95%CI of cooling degree days.
"""
t1 = table_09.to_html()
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))

cap = """
<b> Table 2.</b> <em> Point estimation and 95%CI of 2015 data.</em>
When Temperature Type is 'heat', then the data is the estimation 
and 95%CI of heating degree days.
When Temperature Type is 'cool', then the data is the estimation 
and 95%CI of cooling degree days.
"""
t2 = table_15.to_html()
t2 = t2.rsplit('\n')
t2.insert(1, cap)
tab2 = ''
for i, line in enumerate(t2):
    tab2 += line
    if i < (len(t2) - 1):
        tab2 += '\n'
display(HTML(tab2))

# #### Part b
# We use the same method to estimate the change between 2009 and 2015 data

change_point_estimation_heat = heat_point_estimation_15-heat_point_estimation_09
change_point_estimation_cool = cool_point_estimation_15-cool_point_estimation_09
sd_change_heat = ((heat_rep_sd_09)**2 +(heat_rep_sd_15)**2)**(0.5)
sd_change_cool = ((cool_rep_sd_09)**2 +(cool_rep_sd_15)**2)**(0.5)
change_heat_lwr = change_point_estimation_heat-norm.ppf(0.975)*sd_change_heat
change_heat_upr = change_point_estimation_heat+norm.ppf(0.975)*sd_change_heat
change_cool_lwr = change_point_estimation_cool-norm.ppf(0.975)*sd_change_cool
change_cool_upr = change_point_estimation_cool+norm.ppf(0.975)*sd_change_cool
store_result_change = [change_heat_lwr,change_point_estimation_heat,change_heat_upr,
                       change_cool_lwr,change_point_estimation_cool,change_cool_upr]
table_change = create_table(store_result_change)

# We get the result ```table_change```. And display it,

cap = """
<b> Table 3.</b> <em> Point estimation and 95%CI of 2015 data.</em>
When Temperature Type is 'heat', then the data is the estimation 
and 95%CI of heating degree days.
When Temperature Type is 'cool', then the data is the estimation 
and 95%CI of cooling degree days.
"""
t3 = table_change.to_html()
t3 = t3.rsplit('\n')
t3.insert(1, cap)
tab3 = ''
for i, line in enumerate(t3):
    tab3 += line
    if i < (len(t3) - 1):
        tab3 += '\n'
display(HTML(tab3))

# ### Problem 3
# First we visualize the result of 2009 data in **Problem 2 Part a**.
# And we will draw a bar for the result.

ax_1 = table_09.plot(kind='barh')

#   ### Figure1: Horizon Bar of 2009 Data Estimation

# Then we consider to draw a errorbar of it.

ax_2 = table_09['point estimation'].plot.barh(
    xerr=table_09['upper bound of estimation']-table_09['point estimation'],
    capsize=4)


# ### Figure2: Horizon Errorbar(bar) of 2009 Data Estimation

# Finally, we consider to draw a errorbar with point estimation as a points instead of bar.
# That is to say We draw a errorbar just like the figure in the slide using in the class.
# This is a liitle bit complex, hence we define ```create_errorbar_point``` to reuse the code.

# +
def create_errorbar_point(table):
    ax = table.loc['heat'].reset_index().plot(
        kind = 'scatter',
        x='point estimation',
        y='Census Region',
        marker='s',
        color='red',
    )

    _ = plt.scatter(
        data = table.loc['cool'].reset_index(),
        x='point estimation',
        y='Census Region',
        marker='s',
        color='green',
    )

    _ = plt.errorbar(
        data=table.loc['heat'].reset_index(),
        x='point estimation',
        y='Census Region',
        fmt='None',
        marker='s',
        xerr=table.loc['heat']['upper bound of estimation']-
             table.loc['heat']['point estimation'],
        ecolor='black',
        capsize=4,
    )
    _ = plt.errorbar(
        data=table.loc['cool'].reset_index(),
        x='point estimation',
        y='Census Region',
        fmt='None',
        marker='s',
        xerr=table.loc['cool']['upper bound of estimation']-
             table.loc['cool']['point estimation'],
        ecolor='black',
        capsize=4
    )
    plt.legend(['heat estiamtion','cool estimation'])

ax_3 = create_errorbar_point(table_09)
# -

# ### Figure3: Horizon Errorbar(point) with 95%CI of 2009 Data Estimation

# For 2015 data. We use the exactly same method to get the figure 

ax_4 = table_15.plot(kind='barh') 

#   ### Figure4: Horizon Bar with 95%CI of 2015 Data Estimation

ax_5 = table_15['point estimation'].plot.barh(
    xerr=table_15['upper bound of estimation']-table_15['point estimation'],
    capsize=4)

# ### Figure5: Horizon Errorbar(bar) with 95%CI of 2015 Data Estimation

ax_6 = create_errorbar_point(table_15)

# ### Figure6: Horizon Errorbar(point) with 95%CI of 2015 Data Estimation
#

# For ```change_table``` we get in **Problem 2 part b**. We use the same way to visualize

ax_7 = table_change.plot(kind='barh')

#   ### Figure7: Horizon Bar of Change Estimation Between 2009 and 2015

ax_8 = table_change['point estimation'].plot.barh(
    xerr=table_change['upper bound of estimation']-table_change['point estimation'],
    capsize=4)

#   ### Figure8: Horizon Errorbar(bar) with 95%CI  of Change Estimation Between 2009 and 2015

ax_9 = create_errorbar_point(table_change)

# ### Figure9: Horizon Errorbar(point) with 95%CI  of Change Estimation Between 2009 and 2015


