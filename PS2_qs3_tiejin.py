import numpy as np
import pandas as pd
import time
import re
import warnings

sas_list = ["Demo_G.XPT","DEMO_H.XPT","DEMO_I.XPT","DEMO_J.XPT"]
res = []
for i in range(len(sas_list)):
    name_list = ['SEQN','RIDAGEYR','RIDRETH3','DMDEDUC2','DMDMARTL',
                 'RIDSTATR','SDMVPSU','SDMVSTRA','WTMEC2YR','WTINT2YR']
    df = pd.read_sas(sas_list[i])[name_list].convert_dtypes()
    df['RIDAGEYR'] = df['RIDAGEYR'].astype("int64")
    df.columns = ['UniqueID','Age','Race and Ethnicity','Education',' Marital Status',
                  'Examination Status','Pseudo-PSU','Pseudo-Stratum','MEC weight','Interview weight']
    df.set_index('UniqueID',inplace=True)
    df['cohort'] = i
    res.append(df)
res = pd.concat(res)
res['cohort'] = pd.Categorical(res['cohort'].replace({0:"2011-2012",1:"2013-2014",
                                                    2:"2015-2016",3:"2017-2018"}))
print(res)
res.to_pickle("demographic data.pickle")

# #### Part b
# We do the same thing as **Part a**. However, we do not know the exact characters in OHXxxTC and OHXxxCTC.
# Hence we need to get all these colmuns first. Here we use regular expression to extract the exact names
# which colmuns we need to retain from original data. We create the ```name_list``` which
# is a list to store the colmuns names we want to retain
# and ```columns_name``` is a list to store the literate name we give first.
#

sas_list_part_c = ["OHXDEN_G.XPT","OHXDEN_H.XPT","OHXDEN_I.XPT","OHXDEN_J.XPT"]
res_c = []
name_list = ['SEQN', 'OHDDESTS']
columns_name = ['UniqueID','Dentition Status']
regu_tooth = r'OHX\d\dTC'
regu_coro = r'OHX\d\dCTC'
all_col = pd.read_sas("OHXDEN_G.XPT").columns
for col in all_col:
    if re.search(regu_tooth,col) != None:
        name_list.append(col)
        liter_name = "Tooth_Count_" + col[3:5]
        columns_name.append(liter_name)
for col in all_col:
    if re.search(regu_coro,col) != None:
        name_list.append(col)
        liter_name = "Coronal_Cavities_" + col[3:5]
        columns_name.append(liter_name)

# Use the two list to get the final result

for i in range(len(sas_list_part_c)):
    df = pd.read_sas(sas_list_part_c[i])[name_list].convert_dtypes()
    df.columns = columns_name
    df.set_index('UniqueID',inplace=True)
    df['cohort'] = i
    res_c.append(df)
res_c = pd.concat(res_c)
res_c['cohort'] = pd.Categorical(res_c['cohort'].replace({0:"2011-2012",1:"2013-2014",
                                                    2:"2015-2016",3:"2017-2018"}))
print(res_c)
res_c.to_pickle("oral health dentition data.pickle")

# ### Part c

print(res.shape)
print(res_c.shape)

# We have the case number of **Part a** dataset is 39156 and the case number of **Part b** dataset is 35909