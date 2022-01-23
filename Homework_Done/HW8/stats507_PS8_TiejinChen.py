# # Problem set 8
# ### Question 0
# Here is the script we use to run in the Great Lakes.
# We only change a little bit to the import part so that
# it imports all the module we will use in the whole Problem Set
# while the the script we actually run does not have module it does not need.
# Here we use two cells to import the module we need is because
# if we import tensorflow or sklearn, then we can not run pyspark in jupyter.
# Hence, we adjust the import before we run qs2.
# (So the second cell does not have run number here)

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean
import findspark

from keras import layers,models,losses,optimizers,regularizers
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import multiprocess as mp
from sklearn.ensemble import GradientBoostingRegressor
import time
from sklearn.metrics import mean_squared_error


# Here we revise ```cv_fold```and ```mse_score```(```gb_score```)
# function in ```cv_funcs``` a little bit to fit our data.

def cv_fold(fit, data, idx_train, idx_val, score, label=None):
    """
    Compute score for a cross-validation fold.

    Parameters
    ----------
    fit - callable.
    A callable to fit an estimator to the training data.
    data - dataframe
    Features and labels for training and validation folds.
    idx_train, idx_val - ndarray or slice.
    Indices for subsetting x and y into training and validation folds.
    More accurately, it should be the index of dataframe instead of raw number.
    score - callable
    Function with signature (res, x, y) for scoring validation sample
    predictions from the estimator fit to training data.
    label - string or None, optional.
    An optional label for tracking results during parallel execution.
    The default is None.

    Returns
    -------
    A tuple (label, metrics) where metrics is the object returned by the
    function passed to score.

    """
    # fit model
    res = fit(data.loc[idx_train].values[:,:-1], data.loc[idx_train].values[:,-1])

    # compute score(s)
    metrics = score(res, data.loc[idx_val].values[:,:-1], data.loc[idx_val].values[:,-1])

    # return scores and label
    return ((label, metrics))


# For rest functions in ```cv_funcs```, we remain the same.

def mse_score(res, x_v, y_v):
    """
    Compute cross entropy and accuracy at each stage of boosting model.

    Parameters
    ----------
    res = an object returned by sklearn.GradientBoosting*
    It's fit method should be called. Th object's predict()
    method is called.

    x_v, y_v - Validation features and labels

    Returns
    -------
    A dictionary with entries `Mse` for the validation
    cross mse after each fold.
    """

    # accuracy
    y_hat = res.predict(x_v)
    mse = np.mean((y_hat-y_v)**2)

    return ({ "Mse": mse})


# multiprocess helper functions: ----------------------------------------------
def calculate(func, args):
    """
    Call func(*args) as part of a worker process.

    Parameters
    ----------
    func - callable function
    args - a tuple of positional args for func

    Returns
    -------
    The result of `func(*args)`

    """
    result = func(*args)
    return (result)


def worker(input, output):
    """
    Function run by worker processes.

    Taken from
    <https://docs.python.org/3/library/multiprocessing.html
    #multiprocessing-programming>

    Parameters
    ----------
    input, output - Queues.
    Input and output queues.

    Returns
    -------
    None, called for it's side effects.
    """
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)


def mp_apply(tasks, n_processes=2):
    """
    Compute tasks in parallel.

    Parameters
    ----------
    tasks - list of tuples.
    A list of tasks, each formulated as a tuple with first element a callable
    and second a tuple of positional arguments.
    n_processes - int, optional.
    The number of child processes used to compute the tasks. The default is 2.

    Returns
    -------
    The unordered results of the computed tasks.

    """
    # create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    # submit tasks
    for task in tasks:
        task_queue.put(task)

    # start processes
    for i in range(n_processes):
        mp.Process(target=worker, args=(task_queue, done_queue)).start()

    # get unordered results
    results = []
    for i, task in enumerate(tasks):
        results.append(done_queue.get())

    # stop child processes
    for i in range(n_processes):
        task_queue.put('STOP')

    return (results)


# First, we prepare all the data, And get the whole data set of training data.

np.random.seed(42)
data = pd.read_csv("train.csv")
mater = pd.read_csv("unique_m.csv")
unique_mat = mater['material'].unique()
np.random.shuffle(unique_mat)
train_mat = unique_mat[:int(len(unique_mat)*0.8)]
val_mat = unique_mat[int(len(unique_mat)*0.8):int(len(unique_mat)*0.9)]
test_mat = unique_mat[int(len(unique_mat)*0.9):]
train_index = mater[mater['material'].isin(train_mat)].index.values
val_index = mater[mater['material'].isin(val_mat)].index.values
test_index = mater[mater['material'].isin(test_mat)].index.values

train_data = data.loc[train_index]
val_data = data.loc[val_index]
test_data = data.loc[test_index]
x_val = val_data.values[:,:-1]
y_val = val_data.values[:,-1]
x_test = test_data.values[:,:-1]
y_test = test_data.values[:,-1]
x_train = train_data.values[:,:-1]
y_train = train_data.values[:,-1]

# Next, we prepare the index of 10-fold training set and testing set in the whole traning data.

fold_mat = []
for i in range(9):
    fold_test_mat = train_mat[i*1243:(i+1)*1243]
    fold_train_mat = list(set(train_mat)-set(fold_test_mat))
    fold_mat.append((fold_train_mat,fold_test_mat))
fold_mat.append((train_mat[:9*1243],train_mat[9*1243:]))
fold_index = []
for fold in fold_mat:
    fold_train_index = mater[mater['material'].isin(fold[0])].index.values
    fold_test_index = mater[mater['material'].isin(fold[1])].index.values
    fold_index.append((fold_train_index,fold_test_index))

gdbt = GradientBoostingRegressor(learning_rate=0.25,random_state=2021,n_estimators=390)
n_processes = 5

# Prepare the all the tasks.


cv_tasks = []
for fold, (idx_train, idx_val) in enumerate(fold_index):
    task = (cv_fold,
            (gdbt.fit,
             train_data,
             idx_train,
             idx_val,
             mse_score,
             'gbdt_fold' + str(fold)
            )
           )
    cv_tasks.append(task)

# We find that jupyter notebook cannot run mulitprocessing
# unless we put our function into other files.
# Hence we will not run the following code
# here to make what we upload concisely.
# And it can run in Pycharm or in Great Lakes.

if __name__ == "__main__":
    print('Start: ' + time.ctime())
    result0 = mp_apply(cv_tasks,n_processes)
    print('End: ' + time.ctime())
    avg_mse = 0
    for result in result0:
        avg_mse += result[1]['Mse']/10
    print("10 folds mse:",avg_mse)
    model_res = gdbt.fit(x_train,y_train)
    y_hat_val = model_res.predict(x_val)
    y_hat_test = model_res.predict(x_test)
    mse_val = mean_squared_error(y_hat_val,y_val)
    mse_test = mean_squared_error(y_hat_test,y_test)
    print("validation mse:{:.3f},test mse:{:.3f}".format(mse_val,mse_test))

# The following code is sh script we use


# \#!/usr/bin/bash
#
# \#
#
# \# Author: Tiejin Chen
#
# \# Updated: Dec 03, 2021
#
# \# slurm options: --------------------------------------------------------------
#
# #SBATCH --job-name=tiejin_ps8qs0
#
# #SBATCH --mail-user=tiejin@umich.edu
#
# #SBATCH --mail-type=BEGIN,END
#
# #SBATCH --cpus-per-task=5
#
# #SBATCH --nodes=1
#
# #SBATCH --ntasks-per-node=1
#
# #SBATCH --mem-per-cpu=5GB
#
# #SBATCH --time=10:00
#
# #SBATCH --account=stats507f21_class
#
# #SBATCH --partition=standard
#
# #SBATCH --output=/home/%u/logs/%x-%j-5.log
#
# \# application: ----------------------------------------------------------------
#
# \# modules 
#
# #SBATCH --get-user-env
#
# \# the contents of this script
#
# # cat run-tiejin_qs0.sh
#
# \# run the script
#
# date
#
# # cd /home/tiejin/
#
# python PS8_chentiejin_qs0.py
#
# date
#
# # echo "Done."

# And the usage information we get from the command line is:
#

# | JobID | NCPUS  |  Elapsed | CPUTime | TotalCPU |
# |  ----  | ---- | ---- | ---- | ---- |
# | 29528720 | 5 |   00:04:19 |  00:21:35 |  15:12.804 |
# | 29528720.ba+ | 5 |   00:04:19 |  00:21:35 |  15:12.803 |
# | 29528720.ex+ | 5 |   00:04:19 |  00:21:35 | 00:00:00 |
#

# We can also get the log:

# ```#!/usr/bin/bash
# #
# # Author: Tiejin Chen
# # Updated: Dec 03, 2021
# # 1: -------------------------------------------------------------------------
#
# # slurm options: --------------------------------------------------------------
# #SBATCH --job-name=tiejin_ps8qs0
# #SBATCH --mail-user=tiejin@umich.edu
# #SBATCH --mail-type=BEGIN,END
# #SBATCH --cpus-per-task=5
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --mem-per-cpu=5GB
# #SBATCH --time=10:00
# #SBATCH --account=stats507f21_class
# #SBATCH --partition=standard
# #SBATCH --output=/home/%u/logs/%x-%j-5.log
#
# # application: ----------------------------------------------------------------
#
# # modules 
# #SBATCH --get-user-env
#
# # the contents of this script
# # cat run-tiejin_qs0.sh
#
# # run the script
# date
#
# # cd /home/tiejin/
# python PS8_chentiejin_qs0.py
#
# date
# # echo "Done."
#
# Fri Dec  3 17:15:13 EST 2021
# Start: Fri Dec  3 17:15:15 2021
# End: Fri Dec  3 17:17:59 2021
# 10 folds mse: 117.09980684701303
# validation mse:109.446,test mse:112.227
# Fri Dec  3 17:19:30 EST 2021
# Done.
#

# ### Question 1
# #### Part a
# in this part, we will consider 6 models:
# 1. model with 3 hidden layers
# 2. model with 1 hidden layer
# 3. model with 2 hidden layers
# 4. model with 3 hidden layers and l2 norm
# 5. model with 3 hidden layers and l1 norm
# 6. model with 3 hidden layers and dropout
#
# All the models are trained with ```mse``` loss and we will use ```relu```
# as the activation function. All the model will be optimized by Adam with learning rate is 1e-4.
# We set batch_size to 32 and epochs to 500 for every model.
# For the parameter of normliztion, we will set it to 0.005 for all models.
# And we will set dropout fraction to 0.3.
# Also we only save the best model during 500 epochs
# with least validation mse, and using the least validation mse to get the best model.
#
# According to the requirement,
# we are not required to search best hyper-parameter here
# and the way we choose the best model is to see their performence in validation set.
# There is no reason to do the cross-validation here because
# in the last we will use the whole data to re-train the model
# and get their last model performance in validation set. Cross-validation
# does not do anything helpful in this process. In contrast,
# If we use cross-validation, we will get 10 models in one kind of model.
# And comparing model_list's performance is quite a large and open question.
# Overall, we will directly use whole train data to fit 6 models.
#

model1 = models.Sequential()
model1.add(layers.Dense(256,activation="relu",input_dim=81))
model1.add(layers.Dense(128,activation="relu"))
model1.add(layers.Dense(128,activation="relu"))
model1.add(layers.Dense(1))
model1.compile(
    optimizers.Adam(learning_rate=1e-4),
    losses.mean_squared_error,
    metrics = [losses.mean_squared_error]
)
file_path = "model1.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history1 = model1.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks=[checkpoint])


model2 = models.Sequential()
model2.add(layers.Dense(256,activation="relu",input_dim=81))
model2.add(layers.Dense(1))
model2.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss = losses.mean_squared_error,
    metrics = ['mean_squared_error']
)
file_path = "model2.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history2 = model2.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks = [checkpoint])


model3 = models.Sequential()
model3.add(layers.Dense(256,activation="relu",input_dim=81))
model3.add(layers.Dense(128,activation="relu"))
model3.add(layers.Dense(1))
model3.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss = losses.mean_squared_error,
    metrics = ['mean_squared_error']
)
file_path = "model3.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history3 = model3.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks = [checkpoint])

model4 = models.Sequential()
model4.add(layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l2(0.005),input_dim=81))
model4.add(layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.005)))
model4.add(layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l2(0.005)))
model4.add(layers.Dense(1))
model4.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss = losses.mean_squared_error,
    metrics = ['mean_squared_error']
)
file_path = "model4.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history4 = model4.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks = [checkpoint])

model5 = models.Sequential()
model5.add(layers.Dense(256,activation="relu",kernel_regularizer=regularizers.l1(0.005),input_dim=81))
model5.add(layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l1(0.005)))
model5.add(layers.Dense(128,activation="relu",kernel_regularizer=regularizers.l1(0.005)))
model5.add(layers.Dense(1))
model5.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss = losses.mean_squared_error,
    metrics = ['mean_squared_error']
)
file_path = "model5.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history5 = model5.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks = [checkpoint])

model6 = models.Sequential()
model6.add(layers.Dense(256,activation="relu",input_dim=81))
model6.add(layers.Dropout(0.3))
model6.add(layers.Dense(128,activation="relu"))
model6.add(layers.Dropout(0.3))
model6.add(layers.Dense(128,activation="relu"))
model6.add(layers.Dense(1))
model6.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss = losses.mean_squared_error,
    metrics = ['mean_squared_error']
)
file_path = "model6.hdf5"
checkpoint = ModelCheckpoint(file_path,monitor='val_loss', verbose=1,
                             save_best_only=True,period=1)
train_history6 = model6.fit(x_train,y_train,epochs=500,batch_size = 32,
                            validation_data = (x_val,y_val),callbacks = [checkpoint])

print("model1 best val mse:{}".format(min(train_history1.history['val_loss'])))
print("model2 best val mse:{}".format(min(train_history2.history['val_loss'])))
print("model3 best val mse:{}".format(min(train_history3.history['val_loss'])))
print("model4 best val mse:{}".format(min(train_history4.history['val_loss'])))
print("model5 best val mse:{}".format(min(train_history5.history['val_loss'])))
print("model6 best val mse:{}".format(min(train_history6.history['val_loss'])))

# #### Part b
# As we can see, the best model here is model4 which is 3 hidden layers with l2 norm.
# We load the best model, predict for test set and get the test mse is 
# 136.88799634697085.

best_model = models.load_model("model4.hdf5")
y_hat_test = best_model.predict(x_test)
print(mean_squared_error(y_hat_test,y_test))

# ### Question 2
# We will upload the script we run in cluster,
# we will also run that code here with local pyspark
# (but no hadoop, and a little bit change in the code) here.

spark_home = "F:\spark-3.1.2-bin-hadoop3.2"
python_path = "E:\python\python.exe"
findspark.init(spark_home,python_path)

user_key = pd.read_csv("stats507/user_key.csv")
tran = pd.read_csv("stats507/triangles.csv")
rect = pd.read_csv("stats507/rectangles.csv")

spark = SparkSession \
    .builder \
    .appName('ps8_qs2') \
    .getOrCreate()
tran_res = {}
rect_res = {}

user_key = spark.createDataFrame(user_key)
user_key = user_key.filter(user_key.user == "tiejin")
tran = spark.createDataFrame(tran)
rect = spark.createDataFrame(rect)

my_tran = user_key.join(tran,user_key.key==tran.key,'left')
my_rect = user_key.join(rect,user_key.key==rect.key,'left')

my_tran_area =my_tran.withColumn("area",my_tran['base']*my_tran['height']*0.5)
my_rect_area =my_rect.withColumn("area",my_rect['width']*my_rect['length'])

mean_tran = my_tran_area.select(mean('area')).first()[0]
mean_rect = my_rect_area.select(mean('area')).first()[0]

total_tran = my_tran_area.select('area').rdd.map(lambda x:x[0]).reduce(lambda x,y:x+y)
total_rect = my_rect_area.select('area').rdd.map(lambda x:x[0]).reduce(lambda x,y:x+y)

# +
tran_res['number'] = my_tran_area.select('area').rdd.count()
tran_res['total areas'] = total_tran
tran_res['mean areas'] = mean_tran
rect_res['number'] = my_rect_area.select('area').rdd.count()
rect_res['total areas'] = total_rect
rect_res['mean areas'] = mean_rect

final_dict = {"triangles":tran_res,
              "rectangles":rect_res}
# -

final_df = pd.DataFrame(final_dict)
final_df.to_csv("ps8_q2_tiejin_results.csv")

# After all the code finish, we can get the csv file, and we read it here.

df = pd.read_csv("ps8_q2_tiejin_results.csv",index_col=0).T

df['number'] = df['number'].astype("int")
df['total areas'] = df['total areas'].apply(lambda x:round(x,2))

df




