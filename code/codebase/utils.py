#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ninadkhargonkar
@author: prachetaamaranath
"""
import pandas as pd
import numpy as np 

def clean_preproc_data_real(filePath):
    """
    Creates a numpy array of the data given the csv file

    Creates a pandas dataframe from the given csv file and exports it to a numpy ndarray. 
    Checks to see if any disease has zero marginal probability and removes it from the dataframe. 
    Readjusts the number of diseases, reindexes the diseases and returns the dataframe. 

    Input Assumption:
    All data that is fed into the file is the form of 0, 1 with the first row being the header row 
    stating the number of the disease (indexed from 0)

    Args:
        filePath: Path to the csv file to load the disease data
    Returns:
        A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
    """
    data=pd.read_csv(filePath, error_bad_lines=False, index_col=None)
    data.reset_index(drop=True, inplace=True)

    disease_numbers = data.columns
    col_names = range(0, len(data.columns))
    mappings = dict(zip(disease_numbers, col_names))

    print("The Disease Mappings are: \n", mappings)

    data.rename(columns=mappings, inplace=True)
    for col in data.columns:
        data[col] = data[col].apply(lambda x: int(1) if x >= 1 else int(0))

    return mappings, data

def clean_preproc_data(filePath):
    """
    Creates a numpy array of the data given the csv file

    Creates a pandas dataframe from the given csv file and exports it to a numpy ndarray. 
    Checks to see if any disease has zero marginal probability and removes it from the dataframe. 
    Readjusts the number of diseases, reindexes the diseases and returns the dataframe. 

    Input Assumption:
    All data that is fed into the file is the form of 0, 1 with the first row being the header row 
    stating the number of the disease (indexed from 0)

    Args:
        filePath: Path to the csv file to load the disease data
    Returns:
        A binary (0-1) numpy ndarray with each row corresponding to a particular 
        person's disease prevalence data. 
    """
    data=pd.read_csv(filePath, error_bad_lines=False)
    
    #Check if any disease does not occur in the dataset at all, if so, that disease has to be removed
    counts = np.sum(data, axis=0)
    to_drop = list(counts[counts==0].index)
    if len(to_drop)!=0:
        print("Disease " + str(to_drop) + " do not occur. Removing them to proceed")
        data.drop(columns=to_drop, inplace=True)
        new_index = np.arange(len(data.columns))
        new_index = [str(i) for i in new_index]
        data.columns = new_index

    return data
