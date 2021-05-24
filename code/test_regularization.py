#!/usr/bin/env python3

"""
Created on Wed Mar 3 17:07:13 2021
@author: prachetaamaranath
"""
import numpy as np
import pandas as pd
import itertools, sys, os, time, pickle, pprint
from matplotlib import pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

path_to_codebase = 'codebase/'
sys.path.insert(0, path_to_codebase)

from utils import clean_preproc_data
from extract_features import ExtractFeatures
from mba import marketbasket
from optimizer import Optimizer
from robust_optimizer import Optimizer as Optimizer_robust

# Function to read the probability distribution from the pickled file 
def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile)         # If error, try encoding='latin1'
    return prob[1]

# Function to retreive the trained support predictor
def read_model(filename):
    with open(filename,"rb") as outfile:
        predictor = pickle.load(outfile)
    return predictor

# Function to compute maxent probabilities after optimization
def compute_prob_exact(optobj):
    """
    INPUT:
    -----
    optobj (Optimizer object) : Takes the optimizer object which has access to the parameters after optimization 

    OUTPUT: 
    -------
    maxent_prob (list): List of probabilities obtained after running maxent on the synthetic data 
    maxent_sum_diseases (list): List of length (d+1) containing the disease-cardinalities from maximum entropy distribution
    """
    # List that stores the 2**n vectors 
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
    
    # Returns the 2**n vectors, and sum of x diseases probabilites
    return maxent_prob, maxent_sum_diseases


def calc_maxent(file_num, width_num=None, time_calc=True):
    """
    INPUT:
    -----
    file_num (int): File number for which we are trying different regularization methods
    width_num (int): Index for width calculation 
    time_calc (boolean): If True, calculates time for maxent computation 

    OUTPUT: 
    -------
    Calculate maximum entropy for width as provided and stores the maxent distribution 
    """

    if time_calc == True: 
        sT = time.time() 

    print("######################################################")
    print("FILE NUMBER: ", file_num, " WIDTH NUMBER: ", width_num)
    print("######################################################")
    print() 

    # Step 1: Get parameters for the given file, and calculate the support using the predictor 

    # Read the parameters for the corresponding file number
    df = pd.read_csv('../data/parameters/parameters_1.csv')
    params = df.loc[df['f'] == file_num]                 # Do not go by index if we are skipping some files (incorrect - df.iloc[file_num-1])    
    d = int(params['d'])
    N = int(params['N'])
    pi_s = float(params['pi_s'])

    # Scale the features 
    scaled_d = (d - 9)/(20 - 9)
    scaled_N = (N - 10000)/(60000 - 10000)
    scaled_pi_s = (pi_s - 1.0)/(1.8 - 1.0)

    # Retrieve random forest regressor model, and use it for predicting support
    predictor = read_model('output/support_predictor_1.pickle')
    covariates = np.array([scaled_d, scaled_N, scaled_pi_s]).reshape(1,-1)

    # Convert to polynomial features 
    from sklearn.preprocessing import PolynomialFeatures 
    poly = PolynomialFeatures(degree = 2) 
    support = predictor.predict(poly.fit_transform(covariates))[0]

    # Print the support predicted
    print("d: ", d, " N: ", N, " pi_s: ", pi_s, " Support: ", support)

    # Step 2: Retrieve synthetic data on which we are going to test regularization 
    synthetic_data_file = '../data/synthetic_data_1/synthetic_data_'+str(file_num)+'.csv' 
    
    # Sanity check to see if file exists (Just in case there was an error in generating the file)
    file_flag = os.path.isfile(synthetic_data_file)
    if file_flag == False:
        print("File does not exist, no synthetic data\n")
        return 
    
    # Step 3: Clean synthetic data (read from csv, and transform into the format used for processing)
    cleaneddata = clean_preproc_data(synthetic_data_file)

    # Step 4: Frequent itemset mining and creating of constraints
    support_dict, marginal_vals, two_wayc, three_wayc, four_wayc = marketbasket(cleaneddata, support)
    if len(support_dict)==0 and len(two_wayc)==0 and len(three_wayc)==0 and len(four_wayc)==0:
        print("No associations are available with specified value of support, only MLE constraints")
        total_constraints = 0
    else:
        total_constraints = len(two_wayc)+len(three_wayc)+len(four_wayc)
    
    # Step 5: If the total number of constraints exceeds 300 or any chosen number input here, use forced partitioning
    if total_constraints > 300:
        feats = ExtractFeatures(cleaneddata.values, Mu=7)
    else:
        feats = ExtractFeatures(cleaneddata.values)

    # Step 6: Assign constraints to the relevant variables
    feats.set_two_way_constraints(two_wayc)
    feats.set_three_way_constraints(three_wayc)
    feats.set_four_way_constraints(four_wayc)
    feats.set_supports(support_dict)
    feats.partition_features()
    print("The clusters are:\n", feats.feat_partitions)

    # Calculate total number of constraints to be used in the maximum entropy computation
    num_constraints = len(two_wayc) + len(three_wayc) + len(four_wayc) 
    print("The number of frequent itemset constraints are: \n", num_constraints)

    all_cons = {}
    for c in two_wayc.keys(): 
        all_cons[c] = support_dict[c]
    for c in three_wayc.keys(): 
        all_cons[c] = support_dict[c]
    for c in four_wayc.keys(): 
        all_cons[c] = support_dict[c]
    pp = pprint.PrettyPrinter(indent=4)
    print("The constraints are: ")
    sorted_cons = sorted(all_cons.items(), key=lambda item: item[1], reverse=True)
    pp.pprint(sorted_cons)

    # Step 7: Depending on the width_num parameter, run either regularized maxent or unregularized maxent
    if width_num != 0:
        
        width_dict = {1:1, 2:1e-1, 3:1e-2, 4:1e-3, 5:1e-4, 6:1e-5, 7:1e-6, 8:1e-7, 9:1e-8, 10:1e-9, 11: 10, 12: 100, 13: 1000, 14:10000, 15:100000}
        width = width_dict[width_num]

        # Regularized optimizer with width parameter
        opt_r_w = Optimizer_robust(feats, width)
        soln_opt_r_w = opt_r_w.solver_optimize()
        if soln_opt_r_w == None:
            print('Solution does not converge')
            return 

        # Calculate the maxEnt and empirical probabilities using the optimized parameters
        maxent_r_w, sum_prob_maxent_r_w = compute_prob_exact(opt_r_w)

        print("Maxent Prob: (Regularized, Width = " +str(width)+ ") ", sum_prob_maxent_r_w)
        print()

        # Store the probabilities in the corresponding output file
        outfilename = 'output/reg_expts/synthetic_data_'+str(file_num)+'_w'+str(width_num)+'.pickle'
        with open(outfilename, "wb") as outfile:
            pickle.dump((maxent_r_w, sum_prob_maxent_r_w), outfile)

    elif width_num == 0:
        # Unregularized maxent to compare the divergences
        opt = Optimizer(feats)
        soln_opt = opt.solver_optimize()
        if soln_opt == None:
            print('Solution does not converge')             # Error message when the optimizer does not reach convergence
            return 

        # Calculate the maxEnt and empirical probabilities using the optimized parameters
        maxent_ur, sum_prob_maxent_ur = compute_prob_exact(opt)

        print("Maxent Prob: (Unregularized) ", sum_prob_maxent_ur)
        print()
       
        # Store the probabilities in the corresponding output file
        outfilename = 'output/reg_expts/synthetic_data_'+str(file_num)+'_ur.pickle'
        with open(outfilename, "wb") as outfile:
            pickle.dump((maxent_ur, sum_prob_maxent_ur), outfile)
        
    if time_calc==True:
        eT = time.time()
        print('Computational time for calculating maxent = {} seconds'.format(eT-sT))
    
    print()

if __name__ == '__main__':
    """
    INPUT: 
    ------
    file_num (int): File number for synthetic data 
    width_num (int): Width number from (1-15)
    """
    file_num = int(sys.argv[1])
    width_num = int(sys.argv[2])
    calc_maxent(file_num=file_num, width_num=width_num)