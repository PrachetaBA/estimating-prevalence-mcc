#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ninadkhargonkar
@author: prachetaamaranath
"""

from __future__ import division
from collections import defaultdict, OrderedDict
import operator
import numpy as np
from pyitlib import discrete_random_variable as drv


class ExtractFeatures(object):
	""" Class summary
	Extract the relevant feature pairs from a given numpy data array to form
	the constraints for the maximum-entropy optimization algorithm. Currently it
	has methods to deal with discrete binary data arrays.

	Give extended description of extraction procedure here (see math classes
	implementation in python for reference on documentation)

	Attributes:
		data_arr: A numpy array (binary) for the disease prevalence data
		K: Total number of constraints to find for maxent optimization
		N: Total number of training examples in the dataset
		feats_pairs_dict: Dict to store the the top K feature pairs along with 
			their values to be used for the constraints.
		feat_graph: Dict to store the transitive graph induced by the feature
			pairs in feats_pairs_dict. Adjacency list representation is used.
		feat_partitions: List to store the partitions (connected components)
			found in the feature graph. Made up of lists containing indices for 
			each partition which have the feature(column) numbers.
	"""

	def __init__(self, dataArray, Mu=None):
		self.data_arr = dataArray
		self.N = self.data_arr.shape[0] # Number of training data examples
		
		#order the values for use in piecewise likelihood
		self.two_way_dict = OrderedDict()
		self.three_way_dict = OrderedDict()
		self.four_way_dict = OrderedDict() 
		
		self.feat_graph = {}             
		self.feat_partitions = [] 

		self.Mu = Mu 			# the maximum size of the approximate clusters
		self.supportdict = {}  	# The dictionary holding support values for every constraint 


	def set_two_way_constraints(self, ext_two_way_dict):
		self.two_way_dict = ext_two_way_dict


	def set_three_way_constraints(self, ext_three_way_dict):
		self.three_way_dict = ext_three_way_dict
	

	def set_four_way_constraints(self, ext_four_way_dict):
		self.four_way_dict = ext_four_way_dict

	def set_supports(self, ext_supportdict):
		self.supportdict=ext_supportdict


	def util_add_edges(self, graph, edge_tup):
		# graph is the dictionary for the partition-graph
		for t in edge_tup:
			for t_ot in edge_tup:
				if t != t_ot:
					graph[t].add(t_ot)

	def util_remove_edges(self, graph, edge_tup):
		for t in edge_tup:
			for t_ot in edge_tup:
				if t!=t_ot:
					if t_ot in graph[t]:
						graph[t].remove(t_ot)

   
	def create_partition_graph(self):
		"""Function to create a graph out of the feature pairs (constraints)
		Two nodes (feature indices) have an edge between them if they appear in 
		a constraint together. The graph is an adjacency list representation
		stored in the graph dictionary.
		
		This method sets the class attribute `feat_graph` to the graph dict

		Args: 
			None

		Returns:
			None
		"""
		graph = {}  # undirected graph
		num_feats = self.data_arr.shape[1]

		# init for each node an empty set of neighbors
		for i in range(num_feats):
			graph[i] = set()

		# create adj-list representation of the graph
		
		for tup_2way in self.two_way_dict.keys():
			# Here tup_2way is a tuple of feature indices            
			self.util_add_edges(graph, tup_2way)

		if len(self.three_way_dict) != 0:
			for tup_3way in self.three_way_dict.keys():
				# Here tup_3way is a triplet of feature indices
				self.util_add_edges(graph, tup_3way)
		
		if len(self.four_way_dict) != 0:
			for tup_4way in self.four_way_dict.keys():
				# Here tup_3way is a triplet of feature indices
				self.util_add_edges(graph, tup_4way)
	
		print()
		self.feat_graph = graph
	

	def partition_features(self):        
		"""Function to partition the set of features (for easier computation).
		Partitioning is equivalent to finding all the connected components in 
		the undirected graph of the features indices as their nodes.         
		This method find the partitions as sets of feature indices and stores 
		them in a list of lists with each inner list storing the indices 
		corresponding to a particular partition.

		This method sets the class attribute `feats_partitions` which is list of
		lists containing the partition assignments.

		Args:
			None
		
		Returns:
			None
		"""
		self.create_partition_graph()
		print("Partitioning the feature graph", end=' ')

		def connected_components(neighbors):
			seen = set()
			def component(node):
				nodes = set([node])
				while nodes:
					node = nodes.pop()
					seen.add(node)
					nodes |= neighbors[node] - seen
					yield node
			for node in neighbors:
				if node not in seen:
					yield component(node)

		partitions = []
		print("and Finding the connected components:")
		for comp in connected_components(self.feat_graph):
			partitions.append(list(comp))

		mu = self.Mu

		# If provided mu, then do forced partitioning, else retain clusters as is. 
		if mu != None: 
			# Assert that the max cluster size is set 
			assert(type(mu) == int)

			# Clusters that need to be approximated 
			large_clusters = [cl for cl in partitions if len(cl) > mu]
			# Remaining clusters that are already satisfying the requirement are retained in partitions
			self.feat_partitions=list(filter(lambda x: len(x) <= mu, partitions)) 

			# See if there exist any clusters which have to be broken down 
			if (len(large_clusters)) == 0: 
				self.feat_partitions = partitions
			# Else, begin the forced partitioning algorithm
			else: 
				print("Forced Partitioning Algorithm begins")

				# Step 1: Begin by mapping the edges to the their corresponding cluster indices 
				edge_clusters = defaultdict(list)

				# Create a dictionary that contains all edges corresponding to the large_clusters 
				for key in self.two_way_dict: 
					for idx, l_c in enumerate(large_clusters): 
						if key[0] in l_c: 
							edge_clusters[idx].append(key)

				for key in self.three_way_dict: 
					for idx, l_c in enumerate(large_clusters): 
						if key[0] in l_c: 
							edge_clusters[idx].append(key)

				for key in self.four_way_dict: 
					for idx, l_c in enumerate(large_clusters): 
						if key[0] in l_c: 
							edge_clusters[idx].append(key)

				# Calculate the support/marginal probabilities for every disease and store in a dictionary 
				diseases = np.arange(len(self.data_arr[0]))
				probabilities = np.sum(self.data_arr, axis = 0)/len(self.data_arr)
				marg_prob = dict(zip(diseases, probabilities))

				# Initialize lists that store the approximated clusters 
				approximated = []
				broken_edges = []

				for idx, cluster in enumerate(large_clusters):
					# Initialize T, N, N_i and T_i as in lines 1-4 in the Forced Partitioning algorithm
					T = edge_clusters[idx]
					N_n = [set([x]) for x in cluster]

					# Calculate the deltas for all values of T 
					deltadict = {}
					for edge in T: 
						edge_product = np.prod([marg_prob[x] for x in edge])
						deltadict[edge] = max(self.supportdict[edge]/edge_product, edge_product/self.supportdict[edge])

					sorted_T = sorted(deltadict.keys(), key=lambda x:deltadict[x])

					while len(sorted_T) != 0: 
						edge = sorted_T.pop()
						X = set(edge) # Select the edge with the highest delta value and pop that from the sorted_T list 

						# Calculate N_union 
						N_union = []
						N_indices = []
						for N_j_idx, N_j in enumerate(N_n): 
							if len(X.intersection(N_j))!=0: 
								N_union.extend(N_j)
								N_indices.append(N_j_idx)
						N_union = set(N_union)

						# Calculate index of the set in N_n which has the minimum edge 
						N_min_idx = None
						for N_j_idx in N_indices:
							if min(X) in N_n[N_j_idx]!=0: 
								N_min_idx = N_j_idx

						if len(N_union) <= mu: 
							for N_i in N_indices: 
								if N_i == N_min_idx:
									N_n[N_i] = N_union
								else:
									N_n[N_i] = set()
						else: 
							broken_edges.append(edge)

				N_n = [list(N) for N in N_n if N!=set()] #filter out empty cluster to feed the optimizer
				for sets in N_n:
					approximated.append(sets)    

				self.feat_partitions.extend(approximated)
				
				""" Remove all edges that were broken from the graph, twowaydict, threewaydict and fourwaydict"""
				for edge in broken_edges:
					self.util_remove_edges(self.feat_graph, edge)
					if len(edge)==2:
						del self.two_way_dict[edge]
					if len(edge)==3:
						del self.three_way_dict[edge]
					if len(edge)==4:
						del self.four_way_dict[edge]
				
				"""Re-add edges from two way dict, three way dict and four way dict"""
				for tup_2way in self.two_way_dict.keys():
					self.util_add_edges(self.feat_graph, tup_2way)

				if len(self.three_way_dict) != 0:
					for tup_3way in self.three_way_dict.keys():
						self.util_add_edges(self.feat_graph, tup_3way)

				if len(self.four_way_dict) != 0:
					for tup_4way in self.three_way_dict.keys():
						self.util_add_edges(self.feat_graph, tup_4way)

		else:
			self.feat_partitions = partitions
		return 	