# -*- coding: utf-8 -*-
# # Question 0
# This is *question 0* for [problem set1](https://jbhender.github.io/Stats507/F21/ps/ps1.html) of [Stats507](https://jbhender.github.io/Stats507/F21/).
#
# >Question 0 is about Markdown.
#
# The next question is about the **Fibonnaci sequence**, $F_n = F_1+F_2$. In part **a**, 
# we will define a Python function `fib_rec()`.  
#
# Below is a …
#
# ### Level 3 Header
#
# Next, we can make a bulleted list:
#
# - Item 1
#     - detail 1     
#     + detail 2
# - Item 2
#
# Finally, we can make an enumerated list:
#
#     a. Item 1    
#     b. Item 2  
#     c. Item 3
#
#  
#

# Now begin our quetions. Firstly, we import all the module we need.

import numpy as np
import time
import pandas as pd
from scipy.stats import norm,beta
import warnings


# # Problem 1
# ### Part a
# For part **a**, we define `fib_rec()` to compute $F_n$ with recursive method. In end of block, 
# we print 3 strings to show the accuracy of our function.

# +
def fib_rec(n,a=0,b=1):
    """

    :param n: type int;  present n in Fn, must greater than 0
    :param a: type float or int;  present F_0
    :param b: type float or int; present F_1
    :return: return a fibonacci result. type float or type int.
    we will use int to convert n, if the input n cannot convert to int, there is a error.

    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    if n == 0:
        return a
    if n == 1:
        return b
    return fib_rec(n-1)+fib_rec(n-2)


test_1 = 'F7 equals to 13, while our {function_name} result is {result}'
test_2 = 'F11 equals to 89, while our {function_name} result is {result}'
test_3 = 'F13 equals to 233, while our {function_name} result is {result}'
print(test_1.format(function_name = 'fib_rec',result = fib_rec(7)))
print(test_2.format(function_name = 'fib_rec',result = fib_rec(11)))
print(test_3.format(function_name = 'fib_rec',result = fib_rec(13)))



# -
# ### Part b
#
# For part **b**, we define a Python function `fib_for` to calculate $F_n$ with for loop. 
# we also print stings to ensure the accuracy.

# +
def fib_for(n):
    """

    :param n: present n in Fn, must greater than 0
    :return:  return a fibonacci result. type int
    we will use int to convert n, if the input n cannot convert to int, there is a error.
    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    res = 0
    n1 = 1
    n2 = 0
    if n == 0:
        return 0
    if n == 1:
        return 1
    for i in range(n-1):
        res = n1+n2
        n2 = n1
        n1 = res
    return res
print(test_1.format(function_name = 'fib_for',result = fib_for(7)))
print(test_2.format(function_name = 'fib_for',result = fib_for(11)))
print(test_3.format(function_name = 'fib_for',result = fib_for(13)))



# -

# ### Part c
#
# For part **c**, we define a Python function `fib_whl` to calculate $F_n$ with while loop. 
# we also print stings to ensure the accuracy. 

def fib_whl(n):
    """

    :param n: present n in Fn, must greater than 0
    :return:  return a fibonacci result. type int
    we will use int to convert n, if the input n cannot convert to int, there is a error.
    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    res = 0
    n1 = 1
    n2 = 0
    i = 0
    if n == 0:
        return 0
    if n == 1:
        return 1
    while i < n-1:
        i+=1
        res = n1 + n2
        n2 = n1
        n1 = res
    return res
print(test_1.format(function_name = 'fib_whl',result = fib_whl(7)))
print(test_2.format(function_name = 'fib_whl',result = fib_whl(11)))
print(test_3.format(function_name = 'fib_whl',result = fib_whl(13)))

# ### Part d
#
# For part **d**, we define a Python function `fib_rnd` to calculate $F_n$ with rounding method. 
# we also print stings to ensure the accuracy. 

def fib_rnd(n):
    """

    :param n: present n in Fn, must greater than 0
    :return:  return a fibonacci result. type int
    we will use int to convert n, if the input n cannot convert to int, there is a error.
    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    phi = (1+5**0.5)/2
    return round(phi**n/(5**0.5))
print(test_1.format(function_name = 'fib_rnd',result = fib_rnd(7)))
print(test_2.format(function_name = 'fib_rnd',result = fib_rnd(11)))
print(test_3.format(function_name = 'fib_rnd',result = fib_rnd(13)))

# ### Part e
#
# For part **e**, we define a Python function `fib_flr` to calculate $F_n$ with truncation method. 
# we also print stings to ensure the accuracy. 

def fib_flr(n):
    """

    :param n: present n in Fn, must greater than 0
    :return:  return a fibonacci result. type int
    we will use int to convert n, if the input n cannot convert to int, there is a error.
    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    phi = (1+5**0.5)/2
    return int(phi**n/(5**0.5)+0.5)
print(test_1.format(function_name = 'fib_flr',result = fib_flr(7)))
print(test_2.format(function_name = 'fib_flr',result = fib_flr(11)))
print(test_3.format(function_name = 'fib_flr',result = fib_flr(13)))


# ### Part f
#
# In this part, we define two function `one_compare` and `compare` to help us to create a table. 
# we will use `one_compare` function in `compare` function. And we will get a pd.dataframe at the end.
#
# We set the default number of experiments for every n to 31 in order to get stable estimation.
# And the result unit is second(s).
#
# We run all the experiments with a cpu AMD Ryzen7 5800x.

def one_compare(n,fun,k=31):
    """

    :param n: type int;
    present n in Fn.
    :param fun: type function;
    present one different function to calculate Fn
    :param k: type int;
    present home many times we use fun to calculate Fn.
    :return: type float;
    return the median time of using fun to calculate Fn in k times.(unit second(s))
    """
    time_res = []
    for i in range(k):
        time_point1 = time.perf_counter()
        temp_fib_res = fun(n)
        time_point2 = time.perf_counter()
        time_res.append(time_point2-time_point1)
    time_res = sorted(time_res)
    if k % 2 == 1:
        return time_res[int((k-1)/2)]
    else:
        return (time_res[int(k/2)] +time_res[int((k/2))-1])/2


def compare(n_list,k=31,table=True):
    """

    :param n_list: type list;
    present all the n in Fn we want to calculate.
    :param k: type int;
    present home many times we use fun to calculate Fn for every n in n_list.
    :param table: type bool;
    present whether we need to create a pd.dataframe.
    :return:
    if table is True, the return is a pd.dataframe. Otherwise it is a dict to store all the time data(unit second(s)).
    """
    time_res_all = {}
    for n in n_list:
        time_res_temp = {}
        rec_time = one_compare(n,fib_rec,k)
        for_time = one_compare(n,fib_for,k)
        whl_time = one_compare(n,fib_whl,k)
        rnd_time = one_compare(n,fib_rnd,k)
        flr_time = one_compare(n,fib_flr,k)
        time_res_temp['rec'] = rec_time
        time_res_temp['for'] = for_time
        time_res_temp['whl'] = whl_time
        time_res_temp['rnd'] = rnd_time
        time_res_temp['flr'] = flr_time
        time_res_all[n] = time_res_temp
    if table == False:
        return time_res_all
    elif table == True:
        df_table = pd.DataFrame(time_res_all)
        return df_table
#
n_list = [5,15,25,35,37,38,39,40]
table = compare(n_list)
print(table.to_markdown())


# With the increasing of n, the time of recusive increase exponential. For loop and While loop increase a little, 
# and time of rounding method and truncation method hardly increase.
#
# when n become larger and larger, it is almost impossible for recusive method to compute, 
# Hence I create hand-craft column to form a new table.

fib_rnd(1475)

hand_dict = {}
n_left_list = [1474, 1475]
for n in n_left_list:
    dict_temp = {'rec': "too much time"}
    dict_temp['for'] = f'{one_compare(n, fib_for):.4e}'
    dict_temp['whl'] = f'{one_compare(n, fib_whl):.2e}'
    if n <1475:
        dict_temp['rnd'] = f'{one_compare(n, fib_rnd):.2e}'
        dict_temp['flr'] = f'{one_compare(n, fib_flr):.2e}'
    else:
        dict_temp['rnd'] = 'cannot compute'
        dict_temp['flr'] = 'cannot compute'
    hand_dict[n] = dict_temp
hand_table = pd.DataFrame(hand_dict)
complete_table = pd.concat([table,hand_table],axis=1)
print(complete_table.to_markdown())


# # Question 2
# ### Part a
#
# In part **a**, we define a new Python function `compute_pascal_raw` with input n presents the raw n we want to 
# compute, and return a list with all the result. We calculate the 7-th raw to show the accuracy.

# +
def compute_pascal_raw(n):
    """

    :param n: type int; present row n we want to compute
    :return: type list; a list with the result we calculate. list[i] presents i-th number of n-th row.
    """
    if n < 0 or type(n)!=type(1):
        raise ValueError('n must be an int greater than 0. ')
    res =[1]
    k =1
    for i in range(1,n+1):
        k *= (n+1-i)/i
        res.append(round(k))
    return res

print(compute_pascal_raw(7))


# -

# ### part b
#
# In this part, we define a function `print_pascal` to print the first n row of pascal triangle.  

# +
def print_pascal(n):
    """

    :param n: type int; present first row n we want to print
    :return: no return.
    """
    if n < 0 or type(n)!=type(1):
        raise ValueError('n must be an int greater than 0. ')
    all_line_str = []
    for i in range(n):
        pas_str = ''
        pas_temp = compute_pascal_raw(i)
        for j in pas_temp:
            pas_str += str(j)
            pas_str += ' '
        all_line_str.append(pas_str)
    max_length = len(all_line_str[-1])
    for string in all_line_str:
        print(string.center(max_length,' '))
        
print_pascal(10)


# -

# When n begins to be bigger, the n-th row we print seems a little bit dislocation. Hence we write a new function
# to solve this problem with bigger spacing.

def print_pascal_perfect(n):
    """
    A more accurate version of pascal print. But less beauty.
    :param n: type int; present first row n we want to print
    :return: no return.
    """
    all_line_str = []
    max_number = max(compute_pascal_raw(n))
    max_length = len(str(max_number))
    for i in range(n):
        pas_str = ''
        pas_temp = compute_pascal_raw(i)
        for j in pas_temp:
            for n in range(max_length+2-len(str(j))):
                pas_str += ' '
            pas_str += str(j)
        all_line_str.append(pas_str)
    max_length = len(all_line_str[-1])
    for string in all_line_str:
        print(string.center(max_length,' '))
print_pascal_perfect(10)

# # Question 3
# ### Part a
# In this part ,we define a function `estimate_mean_a` to give the point estimation and 
# confidence interval of input $x$. 
#
# We create a simple x to show the result.
#

# +
def estimate_mean_a(x,confi,string = None):
    """

    :param x: type np.array;
    the data we want to estimate the mean and confidence level.
    Must must be the type that can convert to np.array.
    :param confi: type float;
    present the confidence level we want to estimate. must less than 100 and greater than 0.
    :param string: None type;
    decide whether we output a string or a dict. if string is not None, we output a str.
    :return:
    if string is True, then we return a string, otherwise we return a dict.
    """
    if confi>100 or confi<0:
        raise ValueError('confidence level must between 0 and 100')
    try:
        x = np.array(x)
        np.mean(x)
    except:
        raise TypeError('x must be the type that can convert to np.array or x must be a number sequence')
    x_bar = np.mean(x)
    x_std = np.std(x)
    x_se = x_std/(len(x)**0.5)
    alpha = 1-confi/100
    z = norm.ppf(1-(alpha/2))
    x_1 = x_bar - x_se*z
    x_2 = x_bar + x_se*z
    dict = {'level':confi}
    dict['est'] = x_bar
    dict['lwr'] = x_1
    dict['upr'] = x_2
    if string != None:
        output_string = '{est:.2f}[{level:.0f}%CI:({lwr:.2f},{upr:.2f})]'.format(**dict)
        return output_string
    else:
        return dict

test = [1,2,3,4,2,3]
print(estimate_mean_a(test,95))
print(estimate_mean_a(test,95,string=True))


# -

# ### Part b
# In this part, we create a function `estimate_mean_b` to get the point esimation and
# confidence interval of input $x$. Here, we assumed that $x$ obey a Binomial distribution.
#
# We integrate four differents method to estimate in the function below. And we will 
# show our result with this function in part **c**. We only show the warning in method 0
# and the error when x cannot be converted to np.array here.
#

# +
def estimate_mean_b(x,confi,method=0,string=None):
    """

    :param x: type np.array;
    the data we want to estimate the mean and confidence level.
    Must must be the type that can convert to np.array.
    :param confi: type float;
    present the confidence level we want to estimate. must less than 100.
    :param method: type int or type str;
    present the different method to estimate. Can use int or str to different method.
    this parameter have following options. we use '~' to present same method.
    0(type int) ~ Standard_Bi(type str)
    1(type int) ~ Clopper_Pearson(type str)
    2(type int) ~ Jeffrey(type str)
    3(type int) ~ Agresti_Coull(type str)
    :param string:
    None type;
    decide whether we output a string or a dict. if string is not None, we output a str.
    :return:
    if string is True, then we return a string, otherwise we return a dict.

    """
    all_method = [0,1,2,3,'Standard_Bi','Clopper_Pearson','Jeffrey','Agresti_Coull']
    if method not in all_method:
        raise ValueError('Wrong method parameter.')
    if confi>100 or confi<0:
        raise ValueError('confidence level must between 0 and 100')
    try:
        x = np.array(x)
        np.mean(x)
    except:
        raise TypeError('x must be the type that can convert to np.array or x must be a number sequence')
    if method == 0 or method == 'Standard_Bi':
        p = np.mean(x)
        n = len(x)
        if min(n*p,n*(1-p)) <= 12:
            warnings.warn('min(np,n(1-p)) is less than 12, this is method may not be good enough'
                          , UserWarning)
        x_se = (p*(1-p)/n)**0.5
        alpha = 1 - confi / 100
        z = norm.ppf(1 - (alpha / 2))
        x_1 = p - x_se * z
        x_2 = p + x_se * z
        dict = {'level': confi}
        dict['est'] = p
        dict['lwr'] = x_1
        dict['upr'] = x_2
        if string != None:
            output_string = '{est:.2f}[{level:.0f}%CI:({lwr:.2f},{upr:.2f})]'.format(**dict)
            return output_string
        else:
            return dict

    if method == 1 or method == 'Clopper_Pearson':
        p = np.mean(x)
        n = len(x)
        s = np.sum(x)
        alpha = 1 - confi / 100
        x_1 = beta.ppf(0.5*alpha,s,n-s+1)
        x_2 = beta.ppf(1-0.5*alpha,s+1,n-s)
        dict = {'level': confi}
        dict['est'] = p
        dict['lwr'] = x_1
        dict['upr'] = x_2
        if string != None:
            output_string = '{est:.2f}[{level:.0f}%CI:({lwr:.2f},{upr:.2f})]'.format(**dict)
            return output_string
        else:
            return dict

    if method == 2 or method == 'Jeffrey':
        p = np.mean(x)
        n = len(x)
        s = np.sum(x)
        alpha = 1 - confi / 100
        x_1 = max(0,beta.ppf(0.5*alpha,s+0.5,n-s+0.5))
        x_2 = min(1,beta.ppf(1-0.5*alpha,s+0.5,n-s+0.5))
        dict = {'level': confi}
        dict['est'] = p
        dict['lwr'] = x_1
        dict['upr'] = x_2
        if string != None:
            output_string = '{est:.2f}[{level:.0f}%CI:({lwr:.2f},{upr:.2f})]'.format(**dict)
            return output_string
        else:
            return dict

    if method == 3 or method == 'Agresti_Coull':
        n = len(x)
        s = np.sum(x)
        alpha = 1 - confi / 100
        z = norm.ppf(1 - (alpha / 2))
        n_hat = n + z**2
        p_hat = (s+(z**2)/2)/n_hat
        x_se = (p_hat*(1-p_hat)/n_hat)**0.5
        x_1 = p_hat - x_se * z
        x_2 = p_hat + x_se * z
        dict = {'level': confi}
        dict['est'] = p_hat
        dict['lwr'] = x_1
        dict['upr'] = x_2
        if string != None:
            output_string = '{est:.2f}[{level:.0f}%CI:({lwr:.2f},{upr:.2f})]'.format(**dict)
            return output_string
        else:
            return dict

x = [0,1,0,0]
estimate_mean_b(x,95,method = 'Standard_Bi')
# -

x = ['Just a test for raise an error',1]
estimate_mean_b(x,95,method = 'Standard_Bi')


# ### Part c
# First, we need to create x. Then we use this x as a input to get the different result.
# Finally we get a pd.dataframe to print.

# +
x = [1 if i <42 else 0 for i in range(42+48)]
n_list = [90,95,99]
method_list = ['Norm_theory','Standard_Bi','Clopper_Pearson','Jeffrey','Agresti_Coull']
dict_all = {}
lwr = 'lwr'
upr = 'upr'
for method in method_list:
    dict_all[method] = {}
    if method == 'Norm_theory':
        for n in n_list:
            temp_dict = estimate_mean_a(x,n)
            new_lwr_key = str(n)+'%level_lwr'
            new_upr_key = str(n)+'%level_upr'
            new_width_key = str(n)+'%level_width'
            dict_all[method][new_lwr_key] = f'{temp_dict[lwr]:.3f}'
            dict_all[method][new_upr_key] = f'{temp_dict[upr]:.3f}'
            dict_all[method][new_width_key] = f'{temp_dict[upr]-temp_dict[lwr]:.3f}'
    else:
        for n in n_list:
            temp_dict = estimate_mean_b(x,n,method=method)
            new_lwr_key = str(n)+'%level_lwr'
            new_upr_key = str(n)+'%level_upr'
            new_width_key = str(n)+'%level_width'
            dict_all[method][new_lwr_key] = f"{temp_dict[lwr]:.3f}"
            dict_all[method][new_upr_key] = f'{temp_dict[upr]:.3f}'
            dict_all[method][new_width_key] = f'{temp_dict[upr]-temp_dict[lwr]:.3f}'
table_all = pd.DataFrame(dict_all)
print(table_all)
            
            
        
    
    
# -

# From the table above we can know that:
#
# for 90,95,99% confidence level, Agresti_Coull method produces the interval with the smallest width.
# Jeffrey method produces the inerval with second smallest width. And the next is Norm_theory
# and Standard_Bi. They are the same method. And Clopper_Pearson produced the most biggest width.
#


