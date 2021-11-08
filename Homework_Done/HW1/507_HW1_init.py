import numpy as np
import time
import pandas as pd
from scipy.stats import norm,beta
import warnings


def fib_rec(n):
    """

    :param n: present n in Fn.
    :return: the fibonacci result
    we will use int to convert n, if the input n cannot convert to int, there is a error.
    """
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib_rec(n-1)+fib_rec(n-2)

def fib_for(n):
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

def fib_whl(n):
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

def fib_rnd(n):
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    phi = (1+5**0.5)/2
    return round(phi**n/(5**0.5))

def fib_flr(n):
    try:
        n = int(n)
    except:
        raise TypeError('n must be an int')
    if n < 0:
        raise ValueError('n must greater than 0')
    phi = (1+5**0.5)/2
    return int(phi**n/(5**0.5)+0.5)


def one_compare(n,fun,k=11):
    """

    :param n: type int;
    present n in Fn.
    :param fun: type function;
    present one different function to calculate Fn
    :param k: type int;
    present home many times we use fun to calculate Fn.
    :return: type float;
    return the median time of using fun to calculate Fn in k times.
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


def compare(n_list,k=11,table=True):
    """

    :param n_list: type list;
    present all the n in Fn we want to calculate.
    :param k: type int;
    present home many times we use fun to calculate Fn for every n in n_list.
    :param table: type bool;
    present whether we need to create a pd.dataframe.
    :return:
    if table is True, the return is a pd.dataframe. Otherwise it is a dict to store all the time data.
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
n_list = [5,15,25,30,35,36,37,38,39,40]
# print(compare(n_list))

def compute_pascal_raw(n):
    res =[1]
    k =1
    for i in range(1,n+1):
        k *= (n+1-i)/i
        res.append(round(k))
    return res

def print_pascal(n):
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


def print_pascal_perfect(n):
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

def estimate_mean_a(x,confi,string = None):
    """

    :param x: type np.array;
    the data we want to estimate the mean and confidence level.
    Must must be the type that can convert to np.array.
    :param confi: type float;
    present the confidence level we want to estimate. must less than 100.
    :param string: None type;
    decide whether we output a string or a dict. if string is not None, we output a str.
    :return:
    if string is True, then we return a string, otherwise we return a dict.
    """
    if confi>100 or confi<0:
        raise ValueError('confidence level must between 0 and 100')
    try:
        x = np.array(x)
    except:
        raise TypeError('x must be the type that can convert to np.array')
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


def estimate_mean_b(x,confi,method,string=None):
    """

    :param x: type np.array;
    the data we want to estimate the mean and confidence level.
    Must must be the type that can convert to np.array.
    :param confi: type float;
    present the confidence level we want to estimate. must less than 100.
    :param method: type int or type str;
    present the different method to estimate. Can use int or str to different method.
    this parameter have following options. we use '~' to present same method.
    0(type int) ~ Standard(type str)
    1(type int) ~ Clopper_Pearson(type str)
    2(type int) ~ Jeffrey(type str)
    3(type int) ~ Agresti_Coull(type str)
    :param string:
    None type;
    decide whether we output a string or a dict. if string is not None, we output a str.
    :return:
    if string is True, then we return a string, otherwise we return a dict.

    """
    all_method = [0,1,2,3,'Standard','Clopper_Pearson','Jeffrey','Agresti_Coull']
    if method not in all_method:
        raise ValueError('Wrong method parameter.')
    if confi>100 or confi<0:
        raise ValueError('confidence level must between 0 and 100')
    try:
        x = np.array(x)
    except:
        raise TypeError('x must be the type that can convert to np.array')
    if method == 0 or method == 'Standard':
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
        x_1 = beta(0.5*alpha,s,n-s+1)
        x_2 = beta(1-0.5*alpha,s+1,n-s)
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
        x_1 = max(0,beta(0.5*alpha,s+0.5,n-s+0.5))
        x_2 = min(1,beta(0.5*alpha,s+0.5,n-s+0.5))
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



#print_pascal_perfect(15)
# print(one_compare(25,fib_rec))
print(estimate_mean_a(np.array([1,2,3]),95,1))







