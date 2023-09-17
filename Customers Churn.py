
"""
<Capstone Project>
Self-defined Functions
Written by Group 10
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def num_dist(data, num):
    """
    Plotting the distributions of numeric variables.
    :param data: a data frame.
    :parma num: a list of columns of all numeric variables.
    :return: show the distribution one by one.
    """
    for i in num:
        plt.hist(data[i], color = "#4C7DAB")
        plt.ylabel('Count')
        plt.xlabel(i)
        plt.show()

def detect_outliers(data, threshold):
    """
    Detecting the outliers of numeric variables.
    :param data: a data frame.
    :parma threshold: the number of standard deviations from the mean.
    :return: print a table of column names and the number of outliers.
    """
    table = pd.DataFrame()
    table['Features'] = data.columns
    table['Outliers'] = [sum(np.abs((data[i] - np.mean(data[i])) / np.std(data[i])) > threshold) for i in data.columns]
    print(table)

def trim_outlier(data, trim_list, threshold):
    """
    Capping and flooring the outliers to 'threshold' standard deviations from the mean.
    :param data: a data frame.
    :param trim_list: a list of columns to be trimmed.
    :parma threshold: the number of standard deviations from the mean.
    :return: the trimmed data frame.
    """
    for i in trim_list:
        floor = np.mean(data[i]) - threshold * np.std(data[i])
        cap = np.mean(data[i]) + threshold * np.std(data[i])
        data[i] = np.where(((data[i] - np.mean(data[i])) / np.std(data[i])) < -threshold, floor, data[i])
        data[i] = np.where(((data[i] - np.mean(data[i])) / np.std(data[i])) > threshold, cap, data[i])
    return data

def check_VIF(X):
    """
    Calculating the VIFs of all variables.
    :param X: a data frame.
    :return: print a table of column names and VIFs.
    """
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
