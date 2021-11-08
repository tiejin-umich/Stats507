# # Homework 2
# ### Problem 0
# We have the following code to review.

sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
op = []
for m in range(len(sample_list)):
    li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][0] == sample_list[n][0] and
                    sample_list[m][3] != sample_list[n][3]):
                li.append(sample_list[n])
        op.append(sorted(li, key=lambda dd: dd[3], reverse=True)[0])
res = list(set(op))

# The code wants to do:
#
# for every distinguishing value of tuple[0], it takes out all the tuple with this distinguishing value,and
# reserve the max tuple which means the tuple has the max value of tuple[2].
# Gathering all this kind of tuple and take out the repeated tuple to get result.
#
# We give the following code review:
# 1. The code needs to work. author needs to focos on where and when to indent.
# Also, keep in mind that the index of list in Python starts at 0.
# Hence we need to use 2 to replace 3 in code.
# 2. The code is very easy to read and Pythonic. For example,
# the author uses an anonymous function when using sorted function which is good.
# 3. Iterate over indices only when necessary, else iterate over values.
# It can make the code run more quickly and author might needs to improve
# 4. The style author write is not bad. It keeps every row with reasonable characters.
#
#

# ### Problem 1
# Before we start, we first import all the packages we needs.

import numpy as np
import pandas as pd
import time
import re
import warnings


# Now, we define a python function ```random_tuple``` to create a list with random tuple with 4 parameters.
# And in the end, we use assert to confirm the every element in the list is a tuple.

# +
def random_tuple(n,k=5,low=0,high=9):
    """

    :param n: type int;
    present the number of random tuple we create
    :param k: type int:
    present the number of random int in every tuple we create
    :param low: number type(eg. int or float);
    present the lowest bound the range where we choose
    :param high:
    number type(eg. int or float);
    present the upper bound the range where we choose
    :return: type list;
    :return a list with every list[i] is a tuple with random int value bewteen low to high.
    """
    res = []
    if n < 0 or k < 0:
        raise ValueError('n and k must be greater than 0')
    if type(n) != type(1) or type(k) != type(1):
        raise TypeError('n and k must be int')
    if type(low) == type(0.5):
        low = int(low) +1
    elif type(low) != type(1):
        raise TypeError('low must be a number')
    if type(high) == type(0.5):
        high = int(high)
    elif type(high) != type(1):
        raise TypeError('high must be a number')
    for i in range(n):
        ran_tup = tuple(np.random.choice(range(low,high+1),k))
        res.append(ran_tup)
    return res

test = random_tuple(10)
for tup in test:
    assert type(tup) == tuple
print("all element are tuple")


# -

# ### Problem2

# #### Part a
# We define a function ```class_max_by_key ``` which means gets the max tuple by one position
# from different class to encapsulate the code. Class here can also been seen as a group_by.
# That is to say, we "group_by" the tuple using class, and then we take every tuple
# which has the max value of one column from each group to form a list. Hence besides the data itself,
# we need 2 more parameters. One is ```class_key``` that can been seen as a position which we use
# class_key-th column to group_bp. And the other one is ```sort_key``` that can also been seen as
# a position which we use sort_key-th column to decide which tuple is the max.
#
# It is easy to see that this two parameters should not been the same.
# Or we can just get the input data without repeated tuple.
# And we use 3 sample list to show the accuracy of our function.

# +
def class_max_by_key(list_data,class_key,sort_key):
    """

    :param list_data: type list;
    the data we want to handle
    :param class_key: type int;
    present the position key that class we have.
    :param sort_key: type int;
    represent which position key to sort.
    Here we assumed sort_key cannot equal to class_key
    :return: type list;
    we will return a list containing the max tuple according to the sort_key from different class.
    """
    if class_key == sort_key:
        warnings.warn("class_key should not equal to sort_key")
        return list(set(list_data))
    op = []
    for m in range(len(list_data)):
        li = [list_data[m]]
        for n in range(len(list_data)):
            if (list_data[m][class_key] == list_data[n][class_key] and
                    list_data[m][sort_key] != list_data[n][sort_key]):
                li.append(list_data[n])
        op.append(sorted(li, key=lambda dd: dd[sort_key], reverse=True)[0])
    res = list(set(op))
    return res

sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
print(class_max_by_key(sample_list,0,2))

sample_list2 = [(1, 3, 5), (0, 1, 2), (1, 9, 4),(2, 1, 3),(0,6,1)]
print(class_max_by_key(sample_list2,0,2))

sample_list3 = [(1, 3, 5), (0, 1, 2), (1, 9, 5),(2, 1, 3),(0,6,1),(2,0,7)]
print(class_max_by_key(sample_list3,0,2))


# -

# #### Part b
# With the code review we made, we write a function ```class_max_by_key_improved``` to
# improve the code. We use same 3 sample list to show the accuracy of fucntion.

# +
#
def class_max_by_key_improved(list_data,class_key,sort_key):
    """

    :param list_data: type list;
    the data we want to handle
    :param class_key: type int;
    present the position key that class we have.
    :param sort_key: type int;
    represent which position key to sort.
    Here we assumed sort_key cannot equal to class_key
    :return: type list;
    we will return a list containing the max tuple according to the sort_key from different class.
    """
    if class_key == sort_key:
        warnings.warn("class_key should not equal to sort_key")
        return list(set(list_data))
    op = []
    for tup in list_data:
        temp_li = [tup]
        for tup_2 in list_data:
            if (tup[class_key] == tup_2[class_key] and
                    tup[sort_key] != tup_2[sort_key]):
                temp_li.append(tup_2)
        op.append(sorted(temp_li, key=lambda dd: dd[sort_key], reverse=True)[0])
    return list(set(op))

print(class_max_by_key_improved(sample_list,0,2))
print(class_max_by_key_improved(sample_list2,0,2))
print(class_max_by_key_improved(sample_list3,0,2))

# -

# #### Part c
# We write a function ```class_max_by_key_dict``` to improve the code further with a new core
# algorithm. Using ```dict``` , the built-in Python data structure with the idea of Hashmap,
# we reduce the time complexity of original code $O(n^2)$  to  $O(n)$.
# We use same 3 sample list to show the accuracy of fucntion.

# +
#
def class_max_by_key_dict(list_data,class_key,sort_key):
    """

    :param list_data: type list;
    the data we want to handle
    :param class_key: type int;
    present the position key that class we have.
    :param sort_key: type int;
    represent which position key to sort.
    Here we assumed sort_key cannot equal to class_key
    :return: type list;
    we will return a list containing the max tuple according to the sort_key from different class.
    """
    if class_key == sort_key:
        warnings.warn("class_key should not equal to sort_key")
        return list(set(list_data))
    max_dict = {}
    res_dict = {}
    for tup in list_data:
        if tup[class_key] not in max_dict.keys() or \
                tup[sort_key] > max_dict[tup[class_key]]:
            max_dict[tup[class_key]] = tup[sort_key]
            res_dict[tup[class_key]] = [tup]
            continue
        if tup[sort_key] == max_dict[tup[class_key]]:
            res_dict[tup[class_key]].append(tup)
    res = []
    for res_list in res_dict.values():
        res += res_list
    return res

print(class_max_by_key_dict(sample_list,0,2))
print(class_max_by_key_dict(sample_list2,0,2))
print(class_max_by_key_dict(sample_list3,0,2))
# -

# #### Part d
# We write a function ```time_estimate ``` to get a nice table with the result of
# our Monte Carlo experiments.
# In total, we will raise our n which is the number of tuple in one list from 5 to 1000.
# For every n, we will run  experiment 100 times, and we record the mean of 100 time experiments.
# Finally, in the table, 'standard' presents the code in **Part a**, improved presents the code
# in **Part b** and dict_method presents the code in **Part c**.


# +
def time_estimate(n_list,num):
    """

    :param n_list: type list;
    every element presents the number of tuple in one list.
    :param num: type int;
    for each n, the number of experiment we run.
    The total experiments we run will be num*len(n_list)
    :return: type pd.DataFrame;
    return a dataframe we can show.
    """
    res = {}
    for n in n_list:
        res_stand = 0
        res_improved = 0
        res_dict = 0
        res_temp = {}
        for i in range(num):
            list_experiment = random_tuple(n)
            time1 = time.perf_counter()
            class_max_by_key(list_experiment,0,3)
            time2 = time.perf_counter()
            class_max_by_key_improved(list_experiment,0,3)
            time3 = time.perf_counter()
            class_max_by_key_dict(list_experiment,0,3)
            time4 = time.perf_counter()
            res_stand = (res_stand*i + time2-time1)/(i+1)
            res_improved = (res_improved*i + time3-time2)/(i+1)
            res_dict = (res_dict*i + time4-time3)/(i+1)
        res_temp['standard'] = res_stand
        res_temp['improved'] = res_improved
        res_temp['dict_method'] = res_dict
        res[n] = res_temp
    return pd.DataFrame(res)

print(time_estimate([5,50,100,200,500,1000],100))
# -

# It is easy to see that after improved the code, the time cost is decreasing.
# However they are still in the same order of magnitude. And the code in **Part c** uses
# much less time than others. We can see it that, when n increase from 50 to 100. n has doubled.
# And the time cost of standard and improved increase 4 times which is $n^2$ while dict_method only
# increase 2 times which is $n$. That coincides with the time complexity we said in **Part c**.

# ### Problem 3
# #### Part a
# In the following code, we use ```convert_dtypes()``` method to convert the data which should be
# int but is float due to the existence of $nan$ into int64. And with this method, we do not need
# to deal with the $nan$ value. And we do not know if we need to do this, but we also set the Unique ids
# as the index of all data. We also change the 'cohort' column to categorical data.
# Finally we use ```to_pickle``` method to save the dataframe as the pickle file.
# We only retain the columns we need and give them literate names of course.
# And we do not handle with the categorical data expect cohort.
# Since we do not know the exact meaning of value for other categorical data.


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
