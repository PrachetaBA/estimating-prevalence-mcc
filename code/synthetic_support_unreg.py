#!/usr/bin/env python3
from __future__ import division
"""
Created on Thu Feb  25 22:20:20 2021
@author: prachetaamaranath
"""

"""
Main Requirements: 
mlxtend version = 0.17.2
python = 3.7.3
"""
import pickle, itertools, sys, time, os.path
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

path_to_codebase = 'codebase/'
sys.path.insert(0, path_to_codebase)

from codebase.utils import clean_preproc_data
from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
from codebase.optimizer import Optimizer
from codebase.mba import marketbasket

# Function to read the probability distribution from the pickle file
def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile)                # If error, use encoding='latin1' as option while loading the pickle file
    return prob[1]

# Function to compute maxent and empirical probabilities after optimization
def compute_prob_exact(optobj):
    """
    INPUT:
    -----
    optobj (Optimizer object) : Takes the optimizer object which has access to the parameters after optimization 

    OUTPUT: 
    -------
    maxent_prob (list): List of probabilities obtained after running maxent on the synthetic data 
    maxent_sum_diseases (list): List of length (d+1) containing the disease-cardinalities from maximum entropy distribution
    emp_prob (list): List of length (d+1) containing the disease-cardinalities from synthetic data
    """
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    
    maxent_sum_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    print("Maxent prob: ", maxent_sum_diseases)
    print('Total Probability: ', total_prob)
    print() 

    emp_prob = np.zeros(num_feats + 1)
    for vec in optobj.feats_obj.data_arr:
        j = sum(vec)
        emp_prob[j] += 1
    emp_prob /= optobj.feats_obj.data_arr.shape[0]
    
    return maxent_prob, maxent_sum_diseases, emp_prob

def main(file_num=None, sup_num=None, time_calc=True):
    """
    INPUT:
    ------
    file_num (int): File number of the synthetic dataset 
    support (int): Input support number for each of the file
    
    OUTPUT: 
    -------
    Stores the maximum entropy distribution for all 2^d vectors in the corresponding output file
    """
    if time_calc == True: 
        sT = time.time() 

    print("######################################################")
    print("FILE NUMBER: ", file_num, " SUPPORT NUMBER: ", sup_num)
    print("######################################################")
    print() 
    
    # Generating synthetic data
    synthetic_data_file = '../data/synthetic_data_1/synthetic_data_'+str(file_num)+'.csv' 
    

    # Sanity check to see if file exists (Just in case there was an error in generating the file)
    file_flag = os.path.isfile(synthetic_data_file)
    if file_flag == False:
        print("File does not exist, no synthetic data\n")
        return 

    # Step 1: Clean synthetic data (read from csv, and transform into the format used for processing)
    cleaneddata = clean_preproc_data(synthetic_data_file)

    # Step 2: Choose appropriate support from provided support number
    support_key = {1:0.001, 2:0.002, 3:0.003, 4:0.004, 5:0.005, 6:0.006, 7:0.007, 8:0.008, 9:0.009, 10:0.01}    # Chosen by user 
    support = support_key[sup_num]
    print("The support is: ", support)
    print()

    # Step 3: Use frequent itemset mining (market-basket analysis) to obtain the pairs, triplets and quartets for constraints
    support_dict, marginal_vals, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    if len(support_dict)==0 and len(two_wayc)==0 and len(three_wayc)==0 and len(four_wayc)==0:
        print("No associations are available with specified value of support, only MLE constraints")
        total_constraints = 0
    else:
        total_constraints = len(two_wayc)+len(three_wayc)+len(four_wayc)
    
    # Step 4: If the total number of constraints exceeds 300 or any chosen number input here, use forced partitioning
    if total_constraints > 300:
        feats = ExtractFeatures(cleaneddata.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata.values)

    # Step 5: Assign constraints to the relevant variables
    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)
    feats.set_supports(support_dict)
    feats.partition_features()
    print("The clusters are:\n", feats.feat_partitions)

    # Calculate total number of constraints to be used in the maximum entropy computation
    num_constraints = len(two_wayc) + len(three_wayc) + len(four_wayc) 
    print("The number of frequent itemset constraints are: \n", num_constraints)

    # Step 6: Call on the L-BFGS-B optimizer to find the appropriate theta and run maximum entropy 
    opt = Optimizer(feats)
    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')             # Error message when the optimizer does not reach convergence
        return 

    # Step 7: Calculate the maxEnt and empirical probabilities using the optimized parameters
    maxent, sum_prob_maxent, emp_prob = compute_prob_exact(opt)
    print()
   
    # Step 8: Store the probabilities in the corresponding output file
    outfilename = 'output/support_expts_1/synthetic_data_'+str(file_num)+'_s'+str(sup_num)+'.pickle'

    # Check if directory exists 
    op_directory = 'output/support_expts_1/'
    if not os.path.isdir(op_directory): 
        os.makedirs(op_directory)
        print("Create directory: ", op_directory)

    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp_prob, num_constraints, support), outfile)
    
    if time_calc==True:
        eT = time.time()
        print('Computational time for calculating maxent = {} seconds'.format(eT-sT))
    
    print()
    
if __name__ == '__main__':
    """
    INPUT: 
    ------
    file_num (int): File number for synthetic data 
    sup_num (int): Support number from list of support values 
    """
    file_num = int(sys.argv[1])
    sup_num = int(sys.argv[2])
    main(file_num=file_num, sup_num=sup_num)