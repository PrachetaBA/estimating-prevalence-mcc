#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  25 22:20:20 2021
@author: prachetaamaranath
"""

from multiprocessing import Process
import numpy as np
import pandas as pd
import sys, random, csv

path_to_codebase = 'codebase/'
sys.path.insert(0, path_to_codebase)
import data_generator, data_helper 

'''
Generate synthetic data parameters following the strategy outlined in the README_synthetic.md
Store the parameters in a data file parameters.csv
'''

class Synthetic_object(object):
	
	'''
	Class to hold parameters for synthetic data generation
	'''
	
	def __init__(self, ds_n):
		'''
		INPUT: 
		------
		ds_n (int) : Dataset number (1,2) for generating the support parameters file 
		
		OUTPUT: 
		-------
		Generates the parameter file having values for the parameters for all synthetic data generated along with the support values
		'''
		# Step 1: Read existing parameters from the parameters file
		
		# File where all parameters are stored
		parameter_file = '../data/parameters/parameters_'+str(ds_n)+'.csv'
		df = pd.read_csv(parameter_file)


		support_parameter_file = '../data/parameters/support_parameters_'+str(ds_n)+'.csv'
		# Step 2: Write into new parameter file
		with open(support_parameter_file, "w") as csvFile: 
			first_row = ['f','s','d','N','pi_s','K','theta','w_p','w_e','v']
			csv.writer(csvFile).writerow(first_row)
		csvFile.close()
		
		# Support ranges
		support = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

		# Step 3: For each combination of key_features, generate 10 different datasets having random parameters for the others
		f_counter = 1
		for file in range(df.shape[0]): 
			for i in range(10):
				d = df.iloc[file]['d']
				N = df.iloc[file]['N']
				pi_s = df.iloc[file]['pi_s']
				K = df.iloc[file]['K']										
				v = df.iloc[file]['v']				# Dirichlet
				theta = df.iloc[file]['theta']		# zipfian parameter
				w_p = df.iloc[file]['w_p']			# w_p
				w_e = df.iloc[file]['w_e']			# w_e

				s = support[i]

				# Write parameters onto file
				row = [f_counter, i+1, d, N, pi_s, K, theta, w_p, w_e, v]
				with open(support_parameter_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)
				csvFile.close()

			# Increment file counter
			f_counter += 1

if __name__ == '__main__':
	ds_n = sys.argv[1]
	obj = Synthetic_object(ds_n)