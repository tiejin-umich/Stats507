# # Problem Set 7
# ### Question 0
# We download the data, and using ```train.csv``` as required.
# And We random divide data into 80 percent of train data,
# 10 percent of validation data and 10 percent of test data.

from sklearn.linear_model import ElasticNetCV,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time

data= pd.read_csv("train.csv")
train_data = data.sample(frac=0.8,random_state=2021)
remain_data = data.drop(train_data.index)
val_data = remain_data.sample(frac=0.5,random_state=2021)
test_data = remain_data.drop(val_data.index)


train_data

val_data

test_data

train_data_x = train_data.values[:, :-1]
train_data_y = train_data.values[:, -1]
val_data_x = val_data.values[:, :-1]
val_data_y = val_data.values[:, -1]
test_data_x = test_data.values[:, :-1]
test_data_y = test_data.values[:, -1]

# ### Question1
# In this question, to reduce the influence of random,
# for all the model, we will set ```random_state``` to 2021.
# #### Part a
# We use ```ElasticNetCV``` to choose the best parameters and gets all the result.
# To choose the best model, For ```l1_ratio```, we choose from \[0,0.05,0.1,...,1 \].
# E.g we choose it from 0 to 1 with step 0.05.
# For ```C```, we choose from \[0,0.25,0.5,...,10\].
# E.g we choose it from 0 to 10 with step 0.25.


l1_ratio_list = list(np.arange(0,1.05,0.05))
c_list = list(np.arange(0,10.1,0.25))
model_a = ElasticNetCV(l1_ratio=l1_ratio_list,n_jobs=8,
                      alphas=c_list,cv=10,random_state=2021)
model_a.fit(train_data_x,train_data_y)

a_res = model_a.mse_path_
a_res = a_res.sum(axis=2)
a_res = a_res/10
a_res = np.flip(a_res,axis=1)

# Now we make the result a dataframe,


df_a = pd.DataFrame(a_res,index=l1_ratio_list,columns=c_list)
df_a.index.name = 'l1_ratio'
df_a.columns.name = 'C'

df_a

# And the best parameter for ```C``` is 0 and for ```l1_ratio``` is 0.95.

print(model_a.alpha_,model_a.l1_ratio_)

# #### Part b
# For this part, we will use ```GridSearchCV``` function in sklearn
# to choose the best model. Due to the time limit
# (running more than 30 mins for the following code),
# for ```max_depth```(max depth for one tree in the Randomforest),
# we will choose from 10 to 24 with step 2.
# For ```n_estimators```(the number of trees in randomforest),
# we will choose from the 80 to 135 with step 5.
#
# And since we will use 10-fold cross-validation(which means ```cv=10```),
# we will use 10 threads to run the code.(which mean ```n_jobs=10```)

tree_depth = list(range(10,26,2))
numtree = list(range(80,140,5))
para_b = {"n_estimators" :numtree,
             "max_depth" : tree_depth}
RF_model = RandomForestRegressor(random_state=2021)
model_b = GridSearchCV(RF_model,
                      param_grid=para_b,
                      n_jobs=10,cv=10,scoring='neg_mean_squared_error')
model_b.fit(train_data_x,train_data_y)

df_b = model_b.cv_results_['mean_test_score'].reshape(len(numtree),len(tree_depth))
df_b = np.abs(df_b)
df_b = pd.DataFrame(df_b,index=numtree,columns=tree_depth)
df_b.index.name = "tree_number"
df_b.columns.name = "tree_depth"

df_b

# And we can get the best parameter is ```num_tree=135```
# and ```tree_depth=20``` , the best mse is 91.249.

print(model_b.best_params_,model_b.best_score_)

# #### Part c
# For this part, we also use ```GridSearchCV``` function.
# And with the same reason. we will choose ```n_estimators```(boosting number)
# from 90 to 390 with step 10. And we set  ```learning rate```  to  0.25
# which is greater than the default value that is 0.1,
# because we want learn the model more quickly than default
# due to the time limit. And we do not want the learning rate is too large
# since it will affect the performance of model.

lr = 0.25
boosting_list = list(range(90,400,10))
para_b = {"n_estimators" :boosting_list}
GBDT_model = GradientBoostingRegressor(learning_rate=lr,random_state=2021)
model_c = GridSearchCV(GBDT_model,
                      param_grid=para_b,
                      n_jobs=10,cv=10,scoring='neg_mean_squared_error')
model_c.fit(train_data_x,train_data_y)

plt.plot(boosting_list,np.abs(model_c.cv_results_['mean_test_score']))
plt.show()

# And the best result here is ```boosting round``` is 290 , the best mse is 107.896.

print(model_c.best_params_,model_c.best_score_)

# ### Question 2
# First, we use 3 best model in **Question 1** to get the MSE from validation set.
# for **part b** and ** part c**, we use the function ```GridSearchCV```.
# And in this function, it will refit the best model with the whole dataset.
# Hence, we only need to use attribute ```best_estimator_``` to
# get the model trained on the whole train set with best parameter in 10-fold CV.

model1 = ElasticNet(alpha=0,l1_ratio=0.95,random_state=2021)
model1.fit(train_data_x,train_data_y)
model2 = model_b.best_estimator_
model3 = model_c.best_estimator_
model_list = [model1,model2,model3]
y_list = [model.predict(val_data_x) for model in model_list]
res_dict = {}
model_name = ["ElasticNet","Random Forest", "GBDT"]
mse_list = [mean_squared_error(y,val_data_y) for y in y_list]
for i in range(3):
    res_dict[model_name[i]] = mse_list[i]


res_val_df = pd.DataFrame(res_dict,index=["validation mse result"])
res_val_df

# The best model according to the result is random forest, And we use it to predict the test set.

best_model = model2
y_pre = best_model.predict(test_data_x)
test_mse = mean_squared_error(y_pre,test_data_y)
print(test_mse)

# Get the test mse is 83.820.


