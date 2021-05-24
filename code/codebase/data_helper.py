#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:30:48 2019

@author: roshanprakash
@author: prachetaamaranath
"""
from scipy.stats import binom 
from scipy.special import comb
import scipy.stats as stats
import numpy as np
import pandas as pd 
import pickle, time, random, bisect, math, itertools
from functools import reduce 

class DataHelper:
	
	def __init__(self, eta, d, K, Beta, v, w_p, w_e):
		"""
		A Data Helper class, that creates clusters of diseases and stores information;
		useful for computing the probability of generation of any disease vector.
   
		PARAMS
		------
		- eta (list of length <d>+1) (using the pi_s in data_generator) : 
					the probability of choosing 'n' diseases 
					in the synthetic generator for all values of n where 0<=n<=d
		- d (int) : the total number of possible diseases
		- K (int) : the number of clusters used for grouping the diseases
		- Beta (list) (same as z in data_generator) : the probabilities of choosing each of 
						 the <K> clusters as the primary cluster
						 while generating a disease vector ; 
						 should sum to 1.0, and
						 len(<Beta>) should be equal to <K>
		- v (list) : the probabilities of choosing each of 
						  the <K> clusters while sampling
						  clusters for grouping diseases ; 
						  should sum to 1.0, and len(<v>) should be 
						  equal to <K>
		- w_p (float) : the first binomial's 'p' value in the overlapping case
		- w_e (float) : the second binomial's 'p' value in the overlapping case
		
		RETURNS
		-------
		None
		"""
		assert len(Beta)==K, 'Incorrect number of Beta parameters! Make sure Beta is available for every cluster.'
		assert abs(sum(Beta)-1.0)<=1e-10, 'Invalid Beta parameters! Should be normalized between 0 and 1, and sum to 1.0!'
		self.d = d
		self.K = K
		self.eta = eta
		self.Beta = Beta
		self.v = v
		self.w_p = w_p
		self.w_e = w_e
		self.overlapping_clusters = self.makeClusters(overlap=True)
		self.overlapping_clusters_stats = self.getClustersSummaries(overlap=True)

	def makeClusters(self, overlap=True):
		"""
		Groups the diseases into different clusters.

		PARAMETERS
		----------
		- overlap(bool, default=True) : if True, overlapping clusters will be created

		RETURNS
		-------
		- a dictionary containing the cluster ID as key and the contained disease numbers
		  (0<=k<=K) as values.
		"""
		assert self.d>=self.K, \
		'Reduce the number of clusters. Not possible to have {} clusters'.format(self.K)
		d_idxs = np.arange(self.d)
		redo = True
		while redo==True:
			clusters = {idx:[] for idx in range(self.K)}
			for d_idx in d_idxs:
				if overlap:
					if self.K==2:
						low=2
					else:
						low=1
					# choose 'm', the number of clusters this disease can belong to uniformly
					m = np.random.randint(low=low, high=self.K+1)
				else:
					# choose only one cluster, since every cluster should be disjoint
					m = 1
				# choose 'm' clusters, without replacement, according to v vector
				selections = np.random.choice(np.arange(self.K), size=m, p=self.v, replace=False)
				for k in selections:
					clusters[k].append(d_idx)
			for k in clusters.keys():
				if len(list(clusters[k]))!=0:
					redo = False
				else:
					redo = True
					break
		return clusters

	def getClustersSummaries(self, overlap=True):
		"""
		Gathers important cluster information relative to the entire sample space of 
		diseases.
		
		PARAMETERS
		----------
		- overlap(bool, default=True) : if True, overlapping clusters will be created
		
		RETURNS
		-------
		- a dictionary containing information regarding the A, B, and E metrics for 
		  every cluster.
		
		NOTE: For any cluster,
		- A : the diseases that are exclusive to the cluster.
		- B : the diseases that are contained in the cluster 
			  and in at least one other cluster.
		- E : the diseases that are not in the cluster.
		"""
		if not overlap:
			clusters = self.disjoint_clusters # not defined 
		else:
			clusters = self.overlapping_clusters
		cluster_stats = {}
		A_k = {} # exclusive diseases  
		B_k = {} # overlapping diseases
		E_k = {} # diseases not contained in any cluster 'k'
		for k in range(self.K):
			A_k[k]=[]
			B_k[k]=[]
			E_k[k] = list(np.delete(np.arange(self.d), clusters[k]))
		for d in range(self.d):
			for k in range(self.K):
				if d in clusters[k]:
					exclusive=True
					for k_ in np.delete(np.arange(self.K), k):
						if d in clusters[k_] and exclusive==True:
							B_k[k].append(d)
							exclusive=False # 'd' is no more exclusive to the cluster 'k'
					if exclusive:
						A_k[k].append(d)
		for k in range(self.K):
			cluster_stats[k] = {'A': A_k[k], 'E': E_k[k], 'B': B_k[k]}

		return cluster_stats

	def computeProbability(self, r, overlap=True):
		"""
		Computes the probability of generating a disease vector 'r'.
		
		PARAMETERS
		----------
		- r (list) : a binary vector of size 'd', the number of diseases
		
		RETURNS
		-------
		- the probability of generating the disease vector 'r', according to the synthetic
		  data generation scheme.
		"""
		observation = list(np.argwhere(np.array(r)==1))
		if observation:
			prob = 0.0
			if overlap:
				for k in self.overlapping_clusters.keys():
					p_k = self.Beta[k]
					D = len(observation)
					size = len(self.overlapping_clusters[k])
					Ak = len(self.overlapping_clusters_stats[k]['A'])
					D_Ak = 0
					D_Bk = 0
					D_Dk = 0
					D_Dk_ = 0
					for d_idx in observation:
						d = list(d_idx)[0] # d_idx is a numpy array containing one integer (look np.argwhere)
						if d in self.overlapping_clusters[k]:
							D_Dk+=1
						else:
							D_Dk_+=1
						if d in self.overlapping_clusters_stats[k]['B']:
							D_Bk+=1
						elif d in self.overlapping_clusters_stats[k]['A']:
							D_Ak+=1
					Bk = np.setdiff1d(self.overlapping_clusters[k], self.overlapping_clusters_stats[k]['A']).size
					Ek = np.setdiff1d(np.arange(self.d), self.overlapping_clusters[k]).size 
					d_s = min(size, D) 
					if D_Dk<d_s:
						b = binom.pmf(D_Dk, D, self.w_p)
						if D_Dk_==Ek: # accounts for revisits 
							for i in range(0, D_Dk):
								if self.d-size<D-i:
									b+=binom.pmf(i, D, self.w_p)  
					else:
						b = 1-binom.cdf(D_Dk-1, D, self.w_p)
						if D_Dk_==Ek: # accounts for revisits 
							for i in range(0, d_s+1):
								if self.d-size<D-i:
									b+=binom.pmf(i, D, self.w_p)
					# NOTE :  Revisits (above) happen only when all out_of_cluster diseases are present in D
					d_ = min(D_Dk, Ak)
					if D_Ak<d_:
						b_exc = binom.pmf(D_Ak, D_Dk, self.w_e) 
						if Bk==D_Bk: # accounts for revisits 
							for j in range(0, D_Ak):
								if Bk<D_Dk-j:
									b_exc+=binom.pmf(j, D_Dk, self.w_e)
					else:
						b_exc = 1-binom.cdf(D_Ak-1, D_Dk, self.w_e)
						if Bk==D_Bk: # accounts for revisits 
							for j in range(0, d_+1):
								if Bk<D_Dk-j:
									b_exc+=binom.pmf(j, D_Dk, self.w_e)
					# NOTE :  Revisits (above) happen only when all overlapping diseases are present in D
					p_k*=(b*b_exc*(1/comb(Ak, D_Ak))*(1/comb(Bk, D_Bk))\
							   *(1/comb(Ek, D_Dk_)))
					prob+=p_k
			else:
				for k in self.disjoint_clusters.keys():
					temp = self.Beta[k]
					j = 0
					for d_idx in observation:
						d = list(d_idx)[0]
						if d in self.disjoint_clusters[k]:
							j+=1 # 'j' is the number of diseases that are in D and D_k
					size = len(self.disjoint_clusters[k])
					if j<size:
						b = binom.pmf(j, len(observation), p=self.w_p)
					elif j==size:
						b = 1-binom.cdf(j-1, len(observation), p=self.w_p)
					a = [list(d_idx)[0] for d_idx in observation]
					for i in self.disjoint_clusters[k]:
						if i in a:
							a.remove(i)
					c = np.delete(np.arange(self.d), self.disjoint_clusters[k]).size
					temp*=(b*(1/comb(size, j))*(1/comb(c, len(a))))
					prob+=temp    
		else:
			prob=1.0
		return prob*self.eta[len(observation)]

	# '''
	# ORIGINAL CODE 
	def computeAll(self, overlap=True, timer=False): 
		"""
		Computes the probabilities of generation of all possible disease vectors.
		
		PARAMETERS
		----------
		- timer (bool, default=False) : if True, records the computational time
		
		RETURNS
		-------
		- A list containing probabilities for all possible 2^d disease vectors.
		"""
		probs = []
		marginal_probs = np.zeros(self.d+1)
		total=0.0
		if timer:
			tic = time.time()
		for idx in range(2**self.d):
			b = format(idx, '0{}b'.format(self.d))
			r = [int(j) for j in b]
			p = self.computeProbability(r, overlap=overlap)
			m = sum(r)
			marginal_probs[m]+=p
			probs.append(p)
			total+=probs[-1]
		print('Sum of probabilities = {}'.format(total))
		if timer:
			toc = time.time()
			print('Computational time for {} probabilities = {} seconds'.format(2**self.d, toc-tic))
		return probs,marginal_probs
	# '''

	'''
	# THRESHOLD CHECKS
	def computeAll(self, overlap=True, timer=False): 
		"""
		Computes the probabilities of generation of all possible disease vectors.
		Compute probabilities in reverse order, so that we can fail faster when checking for threshold violations
		
		PARAMETERS
		----------
		- timer (bool, default=False) : if True, records the computational time
		
		RETURNS
		-------
		- A list containing probabilities for all possible 2^d disease vectors.
		"""
		probs = []
		marginal_probs = np.zeros(self.d+1)
		total=0.0
		if timer:
			tic = time.time()
		for idx in range(2**self.d-1, -1, -1):
			b = format(idx, '0{}b'.format(self.d))
			r = [int(j) for j in b]
			p = self.computeProbability(r, overlap=overlap)
			# Incorporate the check for generating probabilities that are less than the threshold that we have set 
			if p < (1/(320*10**6)): 
				return False, False	
			m = sum(r)
			marginal_probs[m]+=p
			probs.append(p)
			total+=probs[-1]
		print('Sum of probabilities = {}'.format(total))
		if timer:
			toc = time.time()
			print('Computational time for {} probabilities = {} seconds'.format(2**self.d, toc-tic))

		probs.reverse()
		return probs,marginal_probs
	'''

	'''
	# VECTORIZED COMPUTATION
	def computeAll(self, overlap=True, timer=False): 
		"""
		Computes the probabilities of generation of all possible disease vectors.
		Compute probabilities in reverse order, so that we can fail faster when checking for threshold violations
		
		PARAMETERS
		----------
		- timer (bool, default=False) : if True, records the computational time
		
		RETURNS
		-------
		- A list containing probabilities for all possible 2^d disease vectors.
		"""
		probs = []
		marginal_probs = np.zeros(self.d+1)
		total=0.0
		if timer:
			tic = time.time()

		# Calculate probability for each vector
		b_combs = list(itertools.product([0,1], repeat=self.d))
		probs = dict(zip(b_combs, list(map(self.computeProbability, b_combs, [overlap]*len(b_combs)))))
		
		# Check threshold
		if np.any(np.array(list(probs.values())) < (1/(320*10**6))): 
			return False, False 

		# Calculate total probability
		total = sum(probs.values())
		print('Sum of probabilities = {}'.format(total))

		# Calculate marginals 
		for k,v in probs.items(): 
			marginal_probs[sum(k)] += v

		if timer:
			toc = time.time()
			print('Computational time for {} probabilities = {} seconds'.format(2**self.d, toc-tic))

		pdist = list(probs.values())
		return pdist, marginal_probs
	'''