#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  25 22:20:20 2021
@author: prachetaamaranath
"""

from multiprocessing import Process
import numpy as np
import sys, random, csv, os

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
		ds_n (int): Dataset number (1,2 or 3) corresponding to the datasets we want to generate
		
		OUTPUT: 
		-------
		Generates the parameter file having values for the parameters for all synthetic data generated
		'''
		# Step 1: Use known combinations of (d, N, pi_s) 
		combinations = []
		
		# Some odd combinations to be included are done manually 
		combinations.append([9, 10000])
		combinations.append([10, 10000])
		combinations.append([10, 20000])
		combinations.append([11, 10000])
		combinations.append([11, 20000])
		combinations.append([11, 40000])

		# Add the rest of the diseases 
		diseases = np.arange(12, 21)
		sizes = [10000, 20000, 40000, 60000]
		for d in diseases: 
			for s in sizes: 
				combinations.append([d, s])

		# Step 2: Append combinations of pi_s to the existing ones
		key_features = [] 
		scale_params = [1.0, 1.2, 1.4, 1.6, 1.8]
		for c in combinations: 
			for sc in scale_params: 
				new_c = c + [sc]
				key_features.append(new_c)

		# Check if folder exists, if not create it. 
		directory = '../data/parameters/'
		if not os.path.isdir(directory): 
			os.makedirs(directory)
			print("Created directory: ", directory)

		# File where all parameters are stored
		parameter_file = '../data/parameters/parameters_'+str(ds_n)+'.csv'
		with open(parameter_file, "w") as csvFile: 
			first_row = ['f','d','N','pi_s','K','theta','w_p','w_e','v']
			csv.writer(csvFile).writerow(first_row)
		csvFile.close()

		# Step 3: For each combination of key_features, generate 10 different datasets having random parameters for the others
		f_counter = 1
		for feature in key_features: 
			for i in range(10): # earlier I had used 20 different values for each combination 
				d = feature[0]
				N = feature[1]
				pi_s = feature[2]
				K = random.choice([1, int(d*(0.25)), int(d*(0.5)), int(d*(0.75))])	# num clusters
				v = np.random.dirichlet(np.ones(K))*1.000000000						# Dirichlet
				theta = random.uniform(a=0.0,b=1.0)									# zipfian parameter
				w_p = random.uniform(a=0.01,b=0.99)									# w_p
				w_e = random.uniform(a=0.75,b=0.95)									# w_e

				# Write parameters onto file
				row = [f_counter, d, N, pi_s, K, theta, w_p, w_e, v]
				with open(parameter_file, "a") as csvFile: 	
					csv.writer(csvFile).writerow(row)
				csvFile.close()

				# Increment file counter
				f_counter += 1

if __name__ == '__main__':
	ds_n = int(sys.argv[1])
	obj = Synthetic_object(ds_n)