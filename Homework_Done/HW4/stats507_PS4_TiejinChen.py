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

# # Problem Set 4
#
# ## Question 0
# Here we choose windows operations(```rolling```) as our topic.
# First, we import all the module we need(including the package we need in Question 1 and Question 2)
#

import numpy as np
import pandas as pd
import time
import re
import warnings
import scipy.stats
from stats507_PS1_TiejinChen import estimate_mean_b
from tqdm import tqdm
import matplotlib.pyplot as plt

# ## Windows operation
# - In the region of data science, sometimes we need to manipulate
# one raw with two raws next to it for every raw.
# - this is one kind of windows operation.
# - We define windows operation as an operation that
# performs an aggregation over a sliding partition of values (from pandas' userguide)
# - Using ```df.rolling``` function to use the normal windows operation

rng = np.random.default_rng(9 * 2021 * 20)
n=5
a = rng.binomial(n=1, p=0.5, size=n)
b = 1 - 0.5 * a + rng.normal(size=n)
c = 0.8 * a + rng.normal(size=n) 
df = pd.DataFrame({'a': a, 'b': b, 'c': c})
print(df)
df['b'].rolling(window=2).sum()

# ## Rolling parameter 
# In ```rolling``` method, we have some parameter to control the method, And we introduce two:
# - center: Type bool; if center is True, Then the result will move to the center in series.
# - window: decide the length of window or the customed window 

df['b'].rolling(window=3).sum()

df['b'].rolling(window=3,center=True).sum()

df['b'].rolling(window=2).sum()

# +
# example of customed window
window_custom = [True,False,True,False,True]
from pandas.api.indexers import BaseIndexer
class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            if self.use_expanding[i]:
                start[i] = 0
                end[i] = i + 1
            else:
                start[i] = i
                end[i] = i + self.window_size
        return start, end
indexer1 = CustomIndexer(window_size=1, use_expanding=window_custom)    
indexer2 = CustomIndexer(window_size=2, use_expanding=window_custom)

df['b'].rolling(window=indexer1).sum()
# -

df['b'].rolling(window=indexer2).sum()

# ## Windows operation with groupby
# - ```pandas.groupby``` type also have windows operation method,
# Hence we can combine groupby and windows operation.
# - we can also use ```apply``` after we use ```rolling```

df.groupby('a').rolling(window=2).sum()


def test_mean(x):
    return x.mean()
df['b'].rolling(window=2).apply(test_mean)

# ## Question1 
# #### Part a
# in part a, we give 2 version of creating data, the first one is
# create data from original sas file, and the second one is just read the data we have prepared.

# +
# create data from original sas file
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
# -
# read from what we have done
res = pd.read_pickle('demographic data.pickle')
res

# #### Part b
#

sas_list_part_c = ["OHXDEN_G.XPT","OHXDEN_H.XPT","OHXDEN_I.XPT","OHXDEN_J.XPT"]
res_c = []
name_list = ['SEQN', 'OHDDESTS']
columns_name = ['UniqueID','ohx_status']
Ds_dict = {1: 'Complete', 2: 'Partial', 3: 'Not Done'}
for i in range(len(sas_list_part_c)):
    df = pd.read_sas(sas_list_part_c[i])[name_list].convert_dtypes()
    df.columns = columns_name
    df.set_index('UniqueID',inplace=True)
    res_c.append(df)
res_c = pd.concat(res_c)

data = pd.merge(res,res_c,how='left',on = 'UniqueID')

#  After we merge the data, we find that if a data with   ```ohx_status == 1``` must have
#  ```exam_status == 2```. Hence when we create categorical variable ```ohx```
#  we only need to consider ```ohx_status``` and do not need to consider ```exam_status```

data[(data['exam_status'] !='Both interviewed and MEC examined')&(data['ohx_status'] == 1)]

# Now we create new variable

data['under_20'] = data['age'].apply(lambda x: 1 if x<20 else 0)

# +
data['education'] = data['education'].cat.add_categories('nan').fillna('nan')

def college(x):
    if x == "Some college or AA degree" or x == "College graduate or above":
        return 1
    elif pd.isnull(x):
        return 0
    else:
        return 0

data['college - with two levels'] = data['education'].map(college)
# -

data = data[['gender','age','under_20','college - with two levels','exam_status','ohx_status']]

data['ohx'] = data['ohx_status'].map(lambda x: 'missing' if pd.isnull(x) or x != 1 else 'complete')

data['under_20'] = pd.Categorical(data['under_20'].replace({0:"No",1:"Yes"}))
data['college - with two levels'] = pd.Categorical(
    data['college - with two levels'].replace({0:"No college/<20",1:"some college/college graduate"}))

data.to_pickle('part b final data.pickle')
data

# Now we just read from pickle file

## Canvas cannot upload XPT. Hence we must write a version of read prepared data directly
data = pd.read_pickle('part b final data.pickle')
data

# #### Part c 
# the number of subjects removed is 1757

delete_data = data[data['exam_status']!='Both interviewed and MEC examined']
delete_data

# the number of remain is 37399

remain_data = data.drop(delete_data.index)
remain_data

# #### Part d
# Firstly, we get the table without ```age``` and p_value

# +
cate_data_list = ['gender','under_20','college - with two levels']

res_cate = []
for col_name in cate_data_list:
    df = remain_data.groupby('ohx')[col_name].value_counts().unstack(level=0).reset_index()
    temp_col = list(df.columns)
    temp_col[0] = 'type'
    df.columns = temp_col
    df.index = [col_name,col_name]
    res_cate.append(df)
# -

res_cate = pd.concat(res_cate,axis=0)

# +
#res_cate= res_cate.set_index([res_cate.index,'type'])
# -

res_cate['percent_comp'] = round(
    100* res_cate['complete']/(res_cate['complete']+res_cate['missing']),2)
res_cate['percent_miss'] = round(
    100*res_cate['missing']/(res_cate['complete']+res_cate['missing']),2)
res_cate

res_cate['complete'] = res_cate['complete'].map(str)+ "("+res_cate['percent_comp'].map(str)+"%)"
res_cate['missing'] = res_cate['missing'].map(str)+ "("+res_cate['percent_miss'].map(str)+"%)"


res_cate[['complete','missing']]

# Now we consider adding age to the table

age_std = round(remain_data.groupby('ohx')['age'].std(),2)
age_mean = round(remain_data.groupby('ohx')['age'].mean(),2)

age_mean,age_std

res_age = pd.DataFrame({'type':'age',
                        'complete':"{}(std:{})".format(age_mean[0],age_std[0]),
                       'missing':"{}(std:{})".format(age_mean[1],age_std[1])},
                                                     index =['age'])

result = pd.concat([res_age,res_cate],axis=0)

result= result.set_index([result.index,'type'])
result[['complete','missing']]

# Now we calculate p-value. We will use ```ttest_ind``` and ```chi2_contingency``` in ```scipy.stats```.
# And in t_test, we will assume the variance is not equal,
# and to get the result of ```chi2_contingency``` with original formula instead of improved formula.
# Hence we will set ```equal_var``` parameter in ```ttest_ind```
# to False and ```correction``` in ```chi2_contingency``` to False.


p_value_result = []
number_regu = r'[0-9.]*' 


complete_data_age = remain_data[remain_data['ohx'] == 'complete']['age']
missing_data_age = remain_data[remain_data['ohx'] == 'missing']['age']


p_value_result.append(scipy.stats.ttest_ind(complete_data_age.values,
                     missing_data_age.values,equal_var=False).pvalue)

# +
total_table = []
for i in range(3):
    one_table = []
    for j in range(1,3):
        temp_data = result.loc[result.index[2*i+j]]
        temp_complete = int(
            re.match(number_regu,temp_data['complete']).group())
        temp_missing = int(
            re.match(number_regu,temp_data['missing']).group())
        temp_table = [temp_complete,temp_missing]
        one_table.append(temp_table)
    total_table.append(np.array(one_table))
for table in total_table:
    p_value_result.append(scipy.stats.chi2_contingency(table,correction=False)[1])
    p_value_result.append(scipy.stats.chi2_contingency(table,correction=False)[1])


    
    
# -

p_value_str = ['{:.3e}'.format(x) for x in p_value_result]
result['p_value'] = p_value_str


# Finally, we get result

result.drop(['percent_comp','percent_miss'],axis=1,inplace=True)
result

# ## Qustion 2
# #### Part a
# We choose 95% confidence level.
# First we calculate the minimal sample number when
# we need to make margin of error less than 0.005,
# e.g the total width of 95% confidence interval of Monte Carlo estimation is less than 0.01

# +
n_list = [100,300,500,1000]
p_list = [0.1,0.25,0.4,0.5]
method_list = ['Standard_Bi','Clopper_Pearson','Jeffrey','Agresti_Coull']
z = scipy.stats.norm.ppf(.975)
num_min = 40000*(z**2)*0.95*0.05
# num_exp_min = []
# for n in n_list:
#     for p in p_list:
#         num_min = 200*z*((n*p*(1-p))**0.5)
#         num_exp_min.append(num_min)
num_min
            
                    
                    
        
        

        
        
        
# -

# Hence, we let the minimal sample number be 7299.
# However, because the Monte Carlo process is not a perfect Normal disturbtion.
# Hence, if we only use 7299 times experiments, we may get the result CI larger than 0.01.
# Here is one example, we use 7299 as experiment number,
# and we find the width of CI of one of the result is greater than 0.01.

data = []
for i in range(int(num_min)+1):
    data.append(np.random.binomial(1,0.1,size=100))
for method in method_list:
    total_right_one_method = 0
    for data_one in data:
        one_result = estimate_mean_b(data_one,confi=95,method=method)
        if one_result['lwr']<=0.1<=one_result['upr']:
            total_right_one_method += 1
    temp = [1 if x<total_right_one_method else 0 for x in range(int(num_min)+1)]
    temp_interval = estimate_mean_b(temp,confi=95)['upr']-estimate_mean_b(temp,confi=95)['lwr']
    print(temp_interval)

# To avoid this problem, we use 1.03 to multiply with the minimal number we calculate.
# This can make most result CI less than 0.01.
# And Now we do our Monte Carlo. Also in the coding,
# we will also calculate what we want in **Part b**

result_confi = []
result_width = []
for n in tqdm(n_list):
    for p in p_list:
        data = []
        for i in range(int(1.03*num_min)):
            data.append(np.random.binomial(1,p,size=n))
        result_one_confi = []
        result_one_width = {}
        for method in method_list:
            total_right_one_method = 0
            result_width_one_method = []
            for data_one in data:
                one_result = estimate_mean_b(data_one,confi=95,method=method)
                result_width_one_method.append(
                    one_result['upr']-one_result['lwr'])
                if one_result['lwr']<=p<=one_result['upr']:
                    total_right_one_method += 1
            result_one_confi.append(total_right_one_method/int(1.03*num_min))
            result_one_width[method] = result_width_one_method
        result_confi.append(result_one_confi)
        result_width.append(result_one_width)

# We make the result a table to show

table_dict = {}
i=0
for n in n_list:
    for p in p_list:
        index = "({},{})".format(n,p)
        np_result = {}
        for j in range(4):
            np_result[method_list[j]] = result_confi[i][j]
        i += 1
        table_dict[index] = np_result
table_show = pd.DataFrame(table_dict)
table_show.T

# Now we draw the plots

X,Y = np.meshgrid(n_list,p_list)
result_confi = np.array(result_confi)
for i in range(4):
    Z=result_confi[:,i].reshape(4,4)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title("Estimated confidence level("+
                 method_list[i]+" Method)")
    ax.set_xlabel('n')
    ax.set_ylabel('p')


# #### Part b
# Since we mentioned before, we calculate what we need in **Part b** .
# Hence in **Part b**, we also use $int(1.03\times7298)$ as the Monte Carlo replications.

result_width_avg = []
for data_all in result_width:
    width_avg = {}
    for method in method_list:
        width_avg[method] = np.nanmean(data_all[method])
    result_width_avg.append(width_avg)

# We get the width result as following

result_width_avg

# Now we draw the plots

for method in method_list:
    data_method = [x[method] for x in result_width_avg]
    data_method = np.array(data_method).reshape(4,4)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, data_method)
    ax.clabel(CS, inline=True, fontsize=15)
    ax.set_title("Estimated Width of Inverval("+
                 method+" Method)")
    ax.set_xlabel('n')
    ax.set_ylabel('p')


# Finally ,we draw a relative plots compare with Clopper_Pearson method.
# Hence, here we do not draw the plots of Clopper_Pearson method

matrix_clop =  [x['Clopper_Pearson'] for x in result_width_avg]
matrix_clop = np.array(matrix_clop).reshape(4,4)
for method in method_list:
    if method != "Clopper_Pearson":
        data_method = [x[method] for x in result_width_avg]
        data_method = np.array(data_method).reshape(4,4)/matrix_clop
        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, data_method)
        ax.clabel(CS, inline=True, fontsize=15)
        ax.set_title("Estimated Width of Inverval("+
                     method+" Method)")
        ax.set_xlabel('n')
        ax.set_ylabel('p')












