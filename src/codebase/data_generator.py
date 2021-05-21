#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 18:25:00 2019

@author: roshanprakash
@author: prachetaamaranath
"""
import numpy as np
import scipy.stats as stats 
from data_helper import *
import csv, time

class DataGenerator(DataHelper):
	
	def __init__(self, pi_s, theta, d, K, v, w_p=None, w_e=None, true_outfile=None):
		"""
		A Data Generator sub-class, that operates under the <DataHelper> superclass.
		This class contains methods to synthetically generate disease data.
   
		PARAMS
		------
		- pi_s(float): Scale parameter for the exponential distribution used for 
					choosing the number of diseases that a given patient will have (n)
		- theta(float) : the skew for the Zipfian distribution ; used while choosing a cluster 
					 from where in-cluster diseases will be sampled
		- d(int) : the total number of possible diseases
		- K(int) : the number of clusters used for grouping the diseases
		- v(list) : the probabilities of choosing each of the clusters while sampling
					   clusters for grouping diseases ; should sum to 1.0, and
					   len(<v>) should be equal to <K>
		- w_p(float) : the first binomial's 'p' value in the overlapping clusters case (use the same parameter
						even if it is the disjoint case)
		- w_e(float) : the second binomial's 'p' value in the overlapping clusters case
		- true_outfile(string) : the file_name for storing the 2^d values of the true distribution 
		- outlier_flag (bool) : The flag which detects if there is an outlier when we are generating the true distribution
	
		RETURNS
		-------
		Generates a synthetic dataset with the given parameters and stores the true probability distribution
		values in the output file. 
		"""
		self.pi_s = pi_s
		self.theta = theta
		self.d = d
		self.K = K
		self.w_p = w_p
		self.w_e = w_e
		self.outlier_flag = False

		# Verify that the cluster affinity distribution adds up to 1
		v = np.array(v) 
		v = v/sum(v)
		assert abs(1-sum(v))<1e-6, 'v does not add to 1.0, Check and try again!'

		# Call the Data Helper class for part of the computation
		super().__init__(eta=self._compute_eta(self.d), d=self.d, K=K, Beta=self._compute_Beta(), 
			v=v, w_p=w_p, w_e=w_e)

		# print clusters to check if they are the same: 
		print("Clusters: " , self.overlapping_clusters)
		# print("Cluster stats: " , self.overlapping_clusters_stats)

		p_vals, marginals = super().computeAll(timer=True, overlap=True)

		# In case of error in generating the probabilities (or if there is any other condition that fails, then it returns false)
		if p_vals == False and marginals == False: 
			self.outlier_flag = True
			return 

		outfilename = true_outfile
		with open(outfilename, "wb") as outfile:
			pickle.dump((p_vals, marginals), outfile)

	def _compute_eta(self, d):
		""" 
		Computes the probabilities of choosing each of 0 to d diseases according to the simulation scheme. 
		
		PARAMETERS
		----------
		- d(int) : the number of diseases
		
		RETURNS
		-------
		- A list of size d+1, containing the probabilities of choosing each of 0 to d diseases.
		"""		
		x = np.arange(0, d+1)
		eta = np.exp(-x/self.pi_s)
		eta /= eta.sum()
		return eta
	
	def _compute_Beta(self):
		""" 
		Computes the probabilities of choosing each of the <K> clusters as the primary cluster
		
		PARAMETERS
		----------
		- None
		
		RETURNS
		-------
		- A list of size <K>, containing the probabilities of choosing each of the clusters.
		"""
		Beta = np.arange(1, self.K+1)**(-self.theta)
		Beta/=Beta.sum()
		return Beta

	def generate_instance(self, overlap=True):
		"""
		Generates a disease vector using one of two sampling schemes.
		
		PARAMETERS
		----------
		 - overlap(bool, default=True) : if True, overlapping clusters will be accessed from 
		   the super class.
		 
		RETURNS
		-------
		- a binary vector of size 'd', the total number of possible diseases, wherein each bit
		  indicates presence or absence of the corresponding disease.
		"""
		# initializations
		D = []
		r = np.zeros(self.d)
		r = r.astype(int)
		# first, choose 'n', from a truncated exponential distribution
		n = np.random.choice(np.arange(self.d+1), p=self.eta)
		# next, choose 'k', from a truncated zipfian distribution
		k = np.random.choice(np.arange(self.K), p=self.Beta)
		if overlap: # scheme for generation from overlapping clusters
			if  n==0:
				return list(r), n
			C_k = max(min(np.random.binomial(n=n, p=self.w_p), len(self.overlapping_clusters[k])), \
						n+len(self.overlapping_clusters[k])-self.d)
			if self.overlapping_clusters_stats[k]['A']: # this cluster contains exclusive diseases
				A_k = max(min(np.random.binomial(n=C_k, p=self.w_e), len(self.overlapping_clusters_stats[k]['A'])), \
						C_k-len(self.overlapping_clusters_stats[k]['B']))
				D.extend(np.random.choice(self.overlapping_clusters_stats[k]['A'], size=A_k, replace=False))
				D.extend(np.random.choice(self.overlapping_clusters_stats[k]['B'], size=C_k-A_k, replace=False))
			else: # this cluster does NOT contain any exclusive diseases
				A_k = 0
				D.extend(np.random.choice(self.overlapping_clusters_stats[k]['B'], size=C_k, replace=False))
			D.extend(np.random.choice(self.overlapping_clusters_stats[k]['E'], size=n-C_k, replace=False))
		else: # scheme for generation from disjoint clusters; not used at all here
			if n==0: 
				return list(r), n
			C_k = max(min(np.random.binomial(n=n, p=self.w_p), len(self.disjoint_clusters[k])), \
						n+len(self.disjoint_clusters[k])-self.d)
			D.extend(np.random.choice(self.disjoint_clusters[k], size=C_k, replace=False))
			D.extend(np.random.choice(self.disjoint_clusters_stats[k]['E'], size=n-C_k, replace=False))
		r[np.array(D)] = 1
		return list(r), n 

# =========================================================================================================================== #   

def run(file_name, N, pi_s, theta, d, K, v, w_p=None, w_e=None, file_name_real=None, overlap=True):
	""" 
	Runs the synthetic data generator to generate instances and saves the generated instances to disk.
	
	PARAMETERS
	----------
	- file_name(str) : the filename to use for saving generated data instances 
	- N(int) : the number of data instances to generate
	- pi_s(float) : the parameter for the truncated exponential distribution
	- theta(float) : the skew for the Zipfian distribution 
	- d(int) : the total number of possible diseases
	- K(int) : the number of clusters used for grouping the diseases
	- v(list) : the probabilities of choosing each of the clusters while sampling
					   clusters for grouping diseases ; should sum to 1.0, and
					   len(<v>) should be equal to <K>
	- w_p(float) : the first binomial's 'p' value in the overlapping clusters case (and disjoint)
	- w_e(float) : the second binomial's 'p' value in the overlapping clusters case
	- file_name_real(string) : the file name for storing the true distribution 
	
	RETURNS
	-------
	- flag of success (True or False)
	"""
	n = np.zeros(d) # Storing the empirical count of all diseases across the patient dataset
	with open(file_name, "w") as csvFile: 
		first_row = list(np.arange(0, d))
		csv.writer(csvFile).writerow(first_row)
		gen_obj = DataGenerator(pi_s=pi_s, theta=theta, d=d, K=K, v=v, w_p=w_p, w_e=w_e, true_outfile=file_name_real)
		if gen_obj.outlier_flag == True: 
			return False
		elif gen_obj.outlier_flag == False: 
			for i in range(N):
				row, no_diseases = gen_obj.generate_instance(overlap=overlap)
				n[no_diseases-1]+=1
				csv.writer(csvFile).writerow(row)
	
	csvFile.close()
	return True

if __name__ == '__main__':
	#example test case 
	run("../../data/synthetic_data_test_new.csv", N=10, pi_s=1.5, 
		theta=1.0, d=5, K=2, v=[0.3, 0.7], w_p=0.6, w_e=0.9, file_name_real="../../output/true_dist_test_new.pickle")