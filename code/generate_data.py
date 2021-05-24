#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  25 22:20:20 2021
@author: prachetaamaranath
"""
from multiprocessing import Process
import numpy as np
import pandas as pd 
import sys, random, csv, re, os

path_to_codebase = 'codebase/'
sys.path.insert(0, path_to_codebase)
import data_generator, data_helper 

'''
Generate synthetic data from the data generation parameters stored in the synthetic_parameters folder
'''

class Synthetic_object(object):
	'''
	Class to read parameters as in the synthetic data parameters file and generate both synthetic data
	as well as data containing the true distribution probabilities. 
	'''

	def __init__(self, ds_n, f):
		'''
		INPUT:
		------
		ds_n (int): dataset number (1,2 or 3)
		f (int): file number 

		OUTPUT: 
		-------
		Generates two data files: 
		1. synthetic_data_f.csv 		# Contains the synthetically generated patient instances 
		2. true_dist_f.csv 				# Contains the true probability distributions 
		'''
		print("###################")
		print("FILE NUMBER: ", f)
		print("###################")

		self.f = int(f) 
		parameter_file = '../data/parameters/parameters_'+str(ds_n)+'.csv'
		df = pd.read_csv(parameter_file)
		params = df.iloc[self.f-1]				

		self.d = int(params['d'])
		self.K = int(params['K']) 
		self.N = int(params['N'])
		self.theta = float(params['theta'])
		self.pi_s = float(params['pi_s'])
		self.w_p = float(params['w_p'])
		self.w_e = float(params['w_e'])

		# Read numpy array and manipulate to produce the array of probabilities
		v_str = params['v']
		v_str = " ".join(re.split("\s+", v_str, flags=re.UNICODE))
		v_str = v_str.rstrip("\n")
		v_str = v_str.replace('[','')
		v_str = v_str.replace(']','')
		v_str = v_str.replace('"','')
		v_str = v_str.replace(' ',',')

		v = v_str.split(',')
		v = [i for i in v if i]
		self.v = [float(x) for x in v]

	def get_synthetic_data(self, file_name_synthetic, file_name_real):
		flag = data_generator.run(file_name_synthetic, N=self.N, pi_s=self.pi_s, theta=self.theta,
			d=self.d, K=self.K, v=self.v, w_p=self.w_p, w_e=self.w_e, file_name_real=file_name_real, overlap=True)
		if flag == True: 
			print("Data Generated successfully!")
			print("Data Helper done!")
		elif flag == False:
			print("Error in data generation process")

	def main(self):
		file_name_real = '../data/true_data_'+str(ds_n)+'/truedist_expt_'+str(self.f)+'.pickle'
		file_name_synthetic = "../data/synthetic_data_"+str(ds_n)+"/synthetic_data_"+str(self.f)+".csv"

		# Check if directories for data exists, if not create it.
		true_data_dir = '../data/true_data_'+str(ds_n)
		if not os.path.isdir(true_data_dir): 
			os.makedirs(true_data_dir)
			print("Created directory: ", true_data_dir)

		synthetic_data_dir = '../data/synthetic_data_'+str(ds_n)
		if not os.path.isdir(synthetic_data_dir): 
			os.makedirs(synthetic_data_dir)
			print("Created directory: ", synthetic_data_dir)

		self.get_synthetic_data(file_name_synthetic, file_name_real)

if __name__ == '__main__':
	# pass all options in the sys argv
	ds_n = sys.argv[1]
	f = sys.argv[2]
	obj = Synthetic_object(ds_n, f)
	obj.main()
