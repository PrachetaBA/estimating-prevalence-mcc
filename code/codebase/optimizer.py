#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ninadkhargonkar
@author: prachetaamaranath
"""

from __future__ import division
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class Optimizer(object):
    """ Class summary
    Solves the maximum-entropy optimization problem when given an object
    from the ExtractFeatures class which contains the feature paritions and 
    the feature pairs for the constraints. Optimization algorithm uses the 
    `fmin_l_bfgs_b` function from scipy for finding an optimal set of params.
    Attributes:
        feats_obj: Object from the ExtractFeatures class. Has the necessary 
            feature partitions and pairs for the optimization algorithm.
        opt_sol: List with length equal to the number of partitions in the 
            feature graph. Stores the optimal parameters (thetas) for each 
            partitions.
        norm_z: List with length equal to the number of partitions in the feature
            graph. Stores the normalization constant for each of partitions (since
            each partition is considered independent of others).
    """

    def __init__(self, features_object):
        # Init function for the class object
        
        self.feats_obj = features_object
        self.opt_sol = None     
        self.norm_z = None

    # Utility function to check whether a tuple (key from constraint dict)
    # contains all the variables inside the given partition.
    def check_in_partition(self, partition, key_tuple):
        flag = True
        for i in key_tuple:
            if i not in partition:
                flag = False
                break
        return flag


    # This function computes the inner sum of the optimization function objective    
    # could split thetas into marginal and specials
    def compute_constraint_sum(self, thetas, rvec, partition):
        """Function to compute the inner sum for a given input vector. 
        The sum is of the product of the theta (parameter) for a particular
        constraint and the indicator function for that constraint and hence the
        sum goes over all the constraints. Note that probability is
        not calculated here. Just the inner sum that is exponentiated
        later.
        Args:
            thetas: list of the maxent paramters
            
            rvec: vector to compute the probability for. Note that it should be
            the 'cropped' version of the vector with respect to the partition
            supplied i.e only those feature indices.
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 

        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints

        # Just the single feature marginal case --> MLE update
        if len(partition) == 1:
            constraint_sum = 0.0
            if rvec[0] == 1:
                constraint_sum += thetas[0]
            
            return constraint_sum

        
        constraint_sum = 0.0
        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict
       
        # Sanity Checks for the partition and the given vector
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}
        
        def check_condition(key, value):
            # key is a tuple of feature indices
            # value is their corresponding required values
            flag = True
            for i in range(len(key)):
                if rvec[findpos[key[i]]] != value[i]:
                    flag = False
                    break
            return flag

        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            constraint_sum += thetas[i] * indicator

        # 2-way constraints
        j = 0
        twoway_offset = num_feats 
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[twoway_offset + j] * indicator
                j += 1

        # 3-way constraints
        j = 0
        threeway_offset = twoway_offset + num_2wayc
        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[threeway_offset + j] * indicator
                j += 1

        # 4-way constraints
        j = 0
        fourway_offset = threeway_offset + num_3wayc
        for key,val in fourway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                constraint_sum += thetas[fourway_offset + j] * indicator
                j += 1

        return constraint_sum



    # This function computes the constraint array    
    # for the entire dataset. Then that array can 
    # be used over and over again
    def compute_data_stats(self, partition):
        """
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 

        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints     

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict
       
        # Sanity Checks for the partition and the given vector
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc 
        data_stats_vector = np.zeros(len_theta)
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}

        N = self.feats_obj.N        
        data_arr = self.feats_obj.data_arr        
        for i in range(N):
            rvec = data_arr[i, partition]
            tmp_arr = self.util_compute_array(rvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos,
                                    num_feats, num_2wayc, num_3wayc, num_4wayc)
            
            data_stats_vector += tmp_arr

        return data_stats_vector


    # This function computes the constraint array for
    # a single vector
    def util_compute_array(self, rvec, partition,
                    twoway_dict, threeway_dict, fourway_dict, findpos,
                    num_feats, num_2wayc, num_3wayc, num_4wayc):

        """Function to compute the inner sum for a given input vector. 
        The sum is of the product of the theta (parameter) for a particular
        constraint and the indicator function for that constraint and hence the
        sum goes over all the constraints. Note that probability is
        not calculated here. Just the inner sum that is exponentiated
        later.
        Args:
            thetas: list of the maxent paramters
            rvec: vector to compute the probability for. Note that it should be
            the 'cropped' version of the vector with respect to the partition
            supplied i.e only those feature indices.
            partition: a list of feature indices indicating that they all belong
            in a single partition and we only need to consider them for now.
        """ 

        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints

        # Just the single feature marginal case --> MLE update
        if len(partition) == 1:
            return rvec[0]  # only the marginal constraint applies here
        
        
        def check_condition(key, value):
            # key is a tuple of feature indices
            # value is their corresponding required values
            flag = True
            for i in range(len(key)):
                if rvec[findpos[key[i]]] != value[i]:
                    flag = False
                    break
            return flag

        len_theta = len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc         
        feat_arr = np.zeros(len_theta)

        # CHECKING WITH 1 since BINARY FEATURES
        # Add up constraint_sum for MARGINAL constraints.
        for i in range(num_feats):
            indicator = 1 if rvec[i] == 1 else 0
            feat_arr[i] = indicator

        # 2-way constraints
        j = 0
        twoway_offset = num_feats
        for key,val in twoway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[twoway_offset + j] = indicator
                j += 1

        # 3-way constraints
        j = 0
        threeway_offset = twoway_offset + num_2wayc
        for key,val in threeway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[threeway_offset + j] = indicator
                j += 1

        # 4-way constraints
        j = 0
        fourway_offset = threeway_offset + num_3wayc
        for key,val in fourway_dict.items():
            if self.check_in_partition(partition, key):                
                indicator = 1 if check_condition(key, val) else 0
                feat_arr[fourway_offset + j] = indicator
                j += 1

        return feat_arr



   # assuming binary features for now.
    def util_constraint_matrix(self, partition):
        """ partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """      
        # thetas is ordered as follows: 
        # (1) all the marginal constraints
        # (2) all the two-way constraints
        # (3) all the three-way constraints
        # (4) all the four-way constraints       

        twoway_dict = self.feats_obj.two_way_dict
        threeway_dict = self.feats_obj.three_way_dict
        fourway_dict = self.feats_obj.four_way_dict       
        
        num_feats = len(partition)  # number of marginal constraints
        num_2wayc = len([1 for k,v in twoway_dict.items() if self.check_in_partition(partition, k)])  # num of 2way constraints for the partition
        num_3wayc = len([1 for k,v in threeway_dict.items() if self.check_in_partition(partition, k)]) # num of 3way constraints for the partition
        num_4wayc = len([1 for k,v in fourway_dict.items() if self.check_in_partition(partition, k)]) # num of 4way constraints for the partition
        len_theta = num_feats + num_2wayc + num_3wayc + num_4wayc
        data_stats_vector = np.zeros(len_theta)
       
        # Reverse lookup hashmap for the indices in the partition
        # Useful to make thetas and the constraint_sum match up consistently
        # rvec's first index corresponds to the first index in the partition
        # with respect to the original vector (before cropping it out for the
        # partiton)
        findpos = {elem:i for i,elem in enumerate(partition)}       

        # Create all permuatations of a vector belonging to that partition
        all_perms = itertools.product([0, 1], repeat=num_feats)
        num_total_vectors = 2**(num_feats)
        constraint_mat = np.zeros((num_total_vectors, len_theta))        
                

        N = self.feats_obj.N

        for i, vec in enumerate(all_perms):
            tmpvec = np.asarray(vec)
            tmp_arr = self.util_compute_array(tmpvec, partition, twoway_dict, 
                                    threeway_dict, fourway_dict, findpos,
                                    num_feats, num_2wayc, num_3wayc, num_4wayc)
            constraint_mat[i,:] = tmp_arr
        
        return constraint_mat

    # normalization constant Z(theta)
    # assuming binary features for now.
    def log_norm_Z(self, thetas, partition, constraint_mat):
        """Computes the log of normalization constant Z(theta) for a given partition
        Uses the log-sum-exp trick for numerical stablility
        Args:
            thetas: The parameters for the given partition
            partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        norm_sum = 0.0       
        num_feats = len(partition)
       
        if num_feats == 1:
            norm_sum = 0.0
            norm_sum = 1 + np.exp(thetas[0])
            return np.log(norm_sum)            
        
        num_total_vectors = 2**(num_feats)
        inner_array = np.dot(constraint_mat, thetas)
        
        log_norm = 0.0
        a_max = np.max(inner_array)
        inner_array -= a_max
        log_norm = a_max + np.log(np.sum(np.exp(inner_array)))

        return log_norm


    # normalization constant Z(theta)
    # assuming binary features for now.
    # Still KEEP it around for len(part) == 1 case
    def binary_norm_Z(self, thetas, partition):
        """Computes the normalization constant Z(theta) for a given partition
        Args:
            thetas: The parameters for the given partition
            partition: List of feature indices indicating that they all belong
            in the same feature-partition.
        """
        norm_sum = 0.0       
        num_feats = len(partition)
       
        if num_feats == 1:
            norm_sum = 1 + np.exp(thetas[0])
            return norm_sum

        # Create all permuatations of a vector belonging to that partition
        all_perms = itertools.product([0, 1], repeat=num_feats)

        for vec in all_perms:
            tmpvec = np.asarray(vec)
            tmp = self.compute_constraint_sum(thetas, tmpvec, partition)
            norm_sum += np.exp(tmp)
        
        return norm_sum


    def solver_optimize(self):
        """Function to perform the optimization
           uses l-bfgsb algorithm from scipy
        """
        parts = self.feats_obj.feat_partitions
        solution = [None for i in parts]
        norm_sol = [None for i in parts]

        for i, partition in enumerate(parts):

            if len(partition) == 1:     # just use the MLE          
                N = self.feats_obj.N
                data_arr = self.feats_obj.data_arr
                feat_col = partition[0]
                mle_count = 0

                for j in range(N):
                    rvec = data_arr[j, feat_col]
                    if rvec == 1:
                        mle_count += 1
                
                if mle_count == 0:
                    print('Zero mle for :', feat_col)
                mle = (mle_count * 1.0)/N                
                theta_opt = np.log(mle/(1-mle))                
                
                # Storing like this to maintain consistency with other
                # partitions optimal solutions
                optimThetas = [theta_opt]
                solution[i] = [optimThetas]  # conv to list to maintain consistency
                norm_sol[i] = self.binary_norm_Z(optimThetas, partition)                
            
            else:         
                datavec_partition = self.compute_data_stats(partition)
                c_matrix_partition = self.util_constraint_matrix(partition)
                len_theta = datavec_partition.shape[0]
                a = np.random.RandomState(seed=1)
                initial_val = a.rand(len_theta)

                def func_objective(thetas):
                    objective_sum = 0.0
                    N = self.feats_obj.N        
                    
                    theta_term = np.dot(datavec_partition, thetas)
                    norm_term = -1 * N * self.log_norm_Z(thetas, partition, c_matrix_partition)
                    objective_sum = theta_term + norm_term

                    return (-1 * objective_sum) # SINCE MINIMIZING IN THE LBFGS SCIPY FUNCTION

                optimThetas = minimize(func_objective, x0=initial_val, method='L-BFGS-B',
                    options={'disp':False, 'maxcor':20, 'ftol':2.2e-10, 'maxfun':500000})

                # Check if the LBFGS-B converges, if doesn't converge, then return error message
                if optimThetas.status != 0:
                    print(optimThetas.message)
                    return None
                
                solution[i] = optimThetas
                norm_sol[i] = np.exp(self.log_norm_Z(optimThetas.x, partition, c_matrix_partition))
                inn_arr = np.dot(c_matrix_partition, optimThetas.x)
                inn_arr = np.exp(inn_arr)
                inn_arr /= norm_sol[i]
                total_prob = np.sum(inn_arr, axis=0)

        self.opt_sol = solution
        self.norm_z = norm_sol
        return (solution, norm_sol)

    def prob_dist(self, rvec):
        """
        Function to compute the probability for a given input vector
        """        
        log_prob = 0.0
        parts = self.feats_obj.feat_partitions
        solution = self.opt_sol
        norm_sol = self.norm_z

        # partition will be a set of indices in the i-th parition        
        for i, partition in enumerate(parts):
            tmpvec = rvec[partition]
            if len(partition)==1:
                term_exp = self.compute_constraint_sum(solution[i][0], tmpvec, partition)
            else:
                term_exp = self.compute_constraint_sum(solution[i].get('x'), tmpvec, partition)

            part_logprob = term_exp - np.log(norm_sol[i])
            log_prob += part_logprob
            part_prob = np.exp(part_logprob)
        
        return np.exp(log_prob)