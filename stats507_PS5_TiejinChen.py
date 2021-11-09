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

# # Problem Set5
#
# ### Question 0
# First we import all the packages we need in whole PS.

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
from itertools import combinations

# Now we fit the model with the following code. we find that even we do not need to manupulate data to categorical since the ```OLS``` function will treat it as categorical automatically

file = 'tooth_growth.feather'
if os.path.exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    #print(tooth_growth.__doc__)
    tg_data = tooth_growth.data
    tg_data.to_feather(file)
tg_data['dose'] = pd.Categorical(tg_data['dose'])
tg_data['len'] = tg_data['len'].apply(np.log)
data = pd.get_dummies(tg_data,columns=['supp']).iloc[:,[0,1,3]]
model1 = sm.OLS.from_formula('len~dose*supp',data=tg_data).fit()
model2 = sm.OLS.from_formula('len~1',data=tg_data).fit()
model3 = sm.OLS.from_formula('len~dose*supp_VC',data=data).fit()
print(model1.summary())
print(model3.summary())

# We have the formula to calculate $R^2: R^2 = 1-\frac{\sum_{i=1}^n(Y_i - \hat{Y_i})^2}{\sum_{i=1}^n (Y_i - \bar{Y_i})^2}=1-\frac{RSS}{TSS}$. And we will use this formula to calculate. And the formula of $Adjust R^2$ is $1-\frac{(1-R^2)(N-1)}{N-p-1}$ where $N$ is the sample size and $p$ is the number of variables.
#
# We use 2 different ways wo get $TSS$ in formula. First, when we do not have any variables, the result of linear model is the $\bar{Y_i}$. Hence the $TSS$ is the $RSS$ of model without any variable. Also, we can use the properites from the model to get $TSS$ directly. We use both method to get the same result.

ssr1 = model1.ssr
ssr2 = model2.ssr
R21 = 1-ssr1/ssr2
R22 = 1-ssr1/model1.centered_tss
print(R21,R22)
Adjust_R21 = 1-(59/54)*(1-R21)
Adjust_R22 = 1-(59/54)*(1-R22)
print(Adjust_R21,Adjust_R22)

# We can see that we get the same result as the $R-squared$ And $Adj. R-squared$  which is 0.794 and 0.775 in the summary of model1 and model3. Also, we use the following code to calculate $RSS$ by ourselves to get the same answer.

y_hat = model1.fittedvalues
residual1 = y_hat-tg_data['len']
rss3 = np.sum(np.square(residual1))
y_hat = model2.fittedvalues
residual1 = y_hat-tg_data['len']
rss4 = np.sum(np.square(residual1))
R23 = 1-rss3/rss4
Adjust_R23 = 1-(59/54)*(1-R23)
print(R23,Adjust_R23)

# ### Question 1
# #### Part A
# In this part, we will use ```OHX01TC``` as our target variable. First, we need to prepare the data.

data1 = pd.read_pickle('oral health dentition data.pickle')
data2 = pd.read_pickle('demographic data.pickle')
data = pd.merge(data1['Tooth_Count_01'],data2,how='left',on = 'UniqueID')


# We cannot deal with the missing value in target variable. Hence we drop these data and let ```OHX01TC``` be a binary Y in order to use logitstic regression.

data.dropna(inplace=True,subset=['Tooth_Count_01'])
data['TC01'] = data['Tooth_Count_01'].apply(lambda x: 1 if x== 2 else 0)

# Now our analysis will only focus on whose ```age``` is greater than 12. And in B-spline part, we will only use knots with age no less than 14.Also we fill all the ```NAN``` value to a new categorical ```missing values```. E.g we think missing values is also a value of variable.

data = data[data['age']>=12]
age = data['age'].unique()
total_age = list(age)
total_age = [x for x in total_age if x>=14]

data['education'] = data['education'].cat.add_categories('missing values')
data['marital_status']=data['marital_status'].cat.add_categories('missing values')
data['education'] = data['education'].fillna('missing values')
data['marital_status'] = data['marital_status'].fillna('missing values')

# And our analysis will only focos on the knots selection with ```degree = 3```. We can know that we need to have 4 knots in total.
#
# if we traverse all the subsets of 4 knots, we need to fit 67 choose 4 model. In my pc, it will take more than 15 hours to pick the model with minimal aic. Hence, Here, we use an idea similar to Coordinate descent. That is to say, we will pick knots one by one. And when we pick the first knot, we will traverse all the subsets of 1 knots to get the best model with 1 knot. And when we pick the second knot, we will traverse all the subsets of 2 knot containing the first knot. And for the third,fourth knot, we have the same idea.

result = []
while len(result) < 4:
    temp_result = 0
    min_aic = 1000000
    for i in total_age:
        poss_knots = result.copy()
        poss_knots.append(i)
        try:
            model = smf.logit('TC01~bs(age,knots=np.array(poss_knots))',data).fit()
            if model.aic < min_aic:
                min_aic = model.aic
                temp_result = i
        except:
            continue
    result.append(temp_result)
    total_age.remove(temp_result)

# And we get the result, the best 4 knots we find is ```[25,79,78,77]``` with the minimal aic is 23188.868 

print(min_aic)
print(result)

# Now we consider how to add demograhic data into the model. First we prepare the data with possible variables to add.

all_col = list(data.columns)
delete_col = ['Tooth_Count_01','age','MEC weight','Interview weight','TC01']
for col in delete_col:
    all_col.remove(col)
all_col

# With only 8 possible variables, we can traverse all the subsets of variables to get the model with minimal aic. Here, we define a function ```choose_best_sub``` to get the best subsets and minimal aic.

formula_str = 'TC01~bs(age,knots=np.array(result)){}'
def choose_best_sub(col_list,data,min_aic_init=1000000):
    length = len(col_list)
    result_col = []
    best_formula = ""
    for i in range(length+1):
        i_combination = list(combinations(col_list,i))
        for poss_col in i_combination:
            format_str = ""
            for col in poss_col:
                format_str += "+"+col
            formula_temp = formula_str.format(format_str)
            try:
                model = smf.logit(formula_temp,data).fit()
            except:
                continue
            if model.aic<min_aic_init:
                min_aic_init = model.aic
                result_col = poss_col
                best_formula = formula_temp
    return result_col,min_aic_init,best_formula
res_demo = choose_best_sub(all_col,data)

# We get the result that the model should contains ```gender```,```race```,```education```,```cohort``` as variables besides b-spline of age. And the minimal aic is 22121.8897.

print(res_demo)

# #### Part b
# First we prepare data ```data_32tc``` for this part

total_tc = []
for col in data1.columns:
    if 'Tooth_Count' in col:
        total_tc.append(col)
data_32tc = pd.merge(
    data1[total_tc],
    data2[['age','gender','race','education','cohort']]
    ,how="left",on="UniqueID")
data_32tc.dropna(inplace=True,subset=total_tc)
data_32tc = data_32tc[data_32tc['age']>=12]
y_formula_str = '{}~bs(age,knots=np.array(result))+gender+race+education+cohort'
for tc in total_tc:
    data_32tc[tc] = data_32tc[tc].apply(lambda x: 1 if x== 2 else 0)
data_32tc['education'] = data_32tc['education'].cat.add_categories('missing values')
data_32tc['education'] = data_32tc['education'].fillna('missing values')

# And we get the final DataFrame using the following codes.

for tc in total_tc:
    y_formula = y_formula_str.format(tc)
    model = smf.logit(y_formula,data_32tc).fit()
    data_32tc[tc+"_probability"] =model.predict()

data_32tc


# #### Part c
# First we prepare the data we need to plot.

age_list = list(data_32tc['age'].unique())
age_res = {}
i = 0
for age_num in age_list:
    age_data = data_32tc[data_32tc['age'] == age_num]
    age_dict = {}
    for tc in total_tc:
        prob = tc+"_probability"
        age_dict[prob] = age_data[prob].mean()
    age_dict['age'] = age_num
    age_res[i] = age_dict
    i += 1


# Then we transform the data to DataFrame. And to plot the line figure, we need to sort our x which is ```age```.

age_res = pd.DataFrame(age_res).T
age_res.sort_values('age',inplace=True)

age_res

# Then we plot 32 subfigure based on Universal Numbering System.

fig = plt.figure()
fig.set_size_inches(16,12)
tooth_name = ['Third Molar','Second Molar','First Molar','Second Bicuspid',
              'First Bicuspid','Cuspid','Lateral Incisor','Central Incisor']
pos_name = ['upper left','upper right','lower right','lower left']
for i in range(len(total_tc)):
    tc = total_tc[i]
    fig.add_subplot(8,4,(i%8)*4+(int(i/8)+1))
    prob = tc+"_probability"
    plt.plot(age_res['age'],age_res[prob])
    if i<8:
        plt.ylabel(tooth_name[i])
    if i%8 == 0:
        plt.title(pos_name[int(i/8)])
fig.tight_layout()

# **Figure 1**: predicted probability that a permanent tooth is present varies with age for each tooth with Universal Numbering System

# ### Question 2

# First we prepare the data for this question.

data_qs2 = data_32tc[['Tooth_Count_01','Tooth_Count_01_probability']]
data_qs2.sort_values('Tooth_Count_01_probability',inplace=True)

# As we spilt the data into 10 groups uniformly, we calculate the bserved proportion of cases with a permanent tooth and expected proportion, and store them in two lists ```obs_result``` and ```expected_result```.

ten_percent = int(25311/10)
obs_result = []
expected_result = []
for i in range(10):
    if i !=9 :
        data_temp = data_qs2[i*ten_percent:(i+1)*ten_percent]
    else:
        data_temp = data_qs2[i*ten_percent:]
    obs = data_temp['Tooth_Count_01'].sum()/len(data_temp)
    expected = data_temp['Tooth_Count_01_probability'].mean()
    obs_result.append(obs)
    expected_result.append(expected)

# And we draw the scatter plot comparing the observed and expected probabilities with a line through the origin with slope 1

plt.scatter(obs_result,expected_result)
plt.xlabel('observed proportion ')
plt.ylabel('expected proportion')
plt.plot([0,0.5],[0,0.5])
plt.show()

# We can see the points in the plot are very close to the line. Although there are some points are not on the line, Only a few points have little distance to the line. Hence we think our model can been said well-calibrated.


