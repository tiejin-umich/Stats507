import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.ensemble import GradientBoostingRegressor
import time
from sklearn.metrics import mean_squared_error


def cv_fold(fit, data, idx_train, idx_val, score, label=None):
    """
    Compute score for a cross-validation fold.

    Parameters
    ----------
    fit - callable.
    A callable to fit an estimator to the training data.
    x, y - ndarray
    Features and labels for training and validation folds.
    idx_train, idx_val - ndarray or slice.
    Indices for subsetting x and y into training and validation folds.
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


def mse_score(res, x_v, y_v):
    """
    Compute cross entropy and accuracy at each stage of boosting model.

    Parameters
    ----------
    res = an object returned by sklearn.GradientBoosting*
    It's fit method should be called. Th object's staged_predict_proba()
    method is called.

    x_v, y_v - Validation features and labels

    Returns
    -------
    A dictionary with entries `ce_val` and `ac_val` for the validation
    cross-entropy and accuracy after each boosting stage.
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



#
# train_data.reset_index(inplace=True,drop=True)
# train_data['fold'] = np.array(train_data.index/1683,dtype=int)
#
#
#
#
# folds = []
# n = train_data.shape[0]
# rows = np.arange(n)
# for fold in range(10):
#     train = np.asarray(train_data['fold'] != fold).nonzero()[0]
#     test = np.asarray(train_data['fold'] == fold).nonzero()[0]
#     folds.append((train, test))

gdbt = GradientBoostingRegressor(learning_rate=0.25,random_state=2021,n_estimators=390)
# x_train = train_data.values[:,:-1]
# y_train = train_data.values[:,-1]
n_processes = 5




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