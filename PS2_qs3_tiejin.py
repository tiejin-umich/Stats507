import numpy as np
import pandas as pd
import time
import re
import warnings


sas_list = ["Demo_G.XPT","DEMO_H.XPT","DEMO_I.XPT","DEMO_J.XPT"]
demo_cat = {
    'gender': {1: 'Male', 2: 'Female'},
    'race': {1: 'Mexican American',
             2: 'Other Hispanic',
             3: 'Non-Hispanic White',
             4: 'Non-Hispanic Black',
             6: 'Non-Hispanic Asian',
             7: 'Other/Multiracial'
             },
    'education': {1: 'Less than 9th grade',
                  2: '9-11th grade (Includes 12th grade with no diploma)',
                  3: 'High school graduate/GED or equivalent',
                  4: 'Some college or AA degree',
                  5: 'College graduate or above',
                  7: 'Refused',
                  9: "Don't know"
                  },
    'marital_status': {1: 'Married',
                       2: 'Widowed',
                       3: 'Divorced',
                       4: 'Separated',
                       5: 'Never married',
                       6: 'Living with partner',
                       77: 'Refused',
                       99: "Don't know"
                       },
    'exam_status': {1: 'Interviewed only',
                    2: 'Both interviewed and MEC examined'
                    }
    }
res = []
for i in range(len(sas_list)):
    name_list = ['SEQN','RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','DMDMARTL',
                 'RIDSTATR','SDMVPSU','SDMVSTRA','WTMEC2YR','WTINT2YR']
    df = pd.read_sas(sas_list[i])[name_list].convert_dtypes()
    df['RIDAGEYR'] = df['RIDAGEYR'].astype("int64")
    df.columns = ['UniqueID','age','gender','race','education','marital_status',
                  'exam_status','Pseudo-PSU','Pseudo-Stratum','MEC weight','Interview weight']
    df.set_index('UniqueID',inplace=True)
    df['cohort'] = i
    res.append(df)
res = pd.concat(res)
res['cohort'] = pd.Categorical(res['cohort'].replace({0:"2011-2012",1:"2013-2014",
                                                    2:"2015-2016",3:"2017-2018"}))
cate_col = res.columns[1:6]
for col_name in cate_col:
    res[col_name] = pd.Categorical(res[col_name].replace(demo_cat[col_name]))

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