#!/usr/bin/env python3
from __future__ import division
"""
Created on Fri Mar  26 09:23:20 2021
@author: prachetaamaranath

INPUT: 
------
None 

OUTPUT: 
-------
This code produces the output of running MaxEnt-MCC on the MEPS data. The raw output are stored in pickle files from which 
the probabilities can be extracted. We also store the constraints in a CSV file in the output folder. 
---------------------------------------------------------------------------------------------------------------------------
meps_prev_cons.csv 
    Stores the constraints output by the FP-Growth algorithm to be used by maxEnt

meps_prev_support.csv
    Stores the support for each constraint in the FP-Growth algorithm to be used by maxEnt

meps_prev_zeroemp.pickle
    python dictionary containing all chronic conditions where the empirical prevalence is 0. 

meps_prevalence_output.pickle
    result-tuple = (maxent, sum_prob_maxent, emp, sum_prob_emp, num_constraints, support) where 
    maxent (list): list of 2^d probabilities obtained after running maxent on the MEPS data 
    sum_prob_maxent (list): list of length (d+1) containing the disease-cardinalities from maximum entropy distribution
    emp (list): List of 2^d empirical probabilities obtained from the MEPS data
    sum_prob_emp (list): list of length (d+1) containing the disease-cardinalities from the MEPS data
    num_constraints (int): number of FP-Growth constraints included
    support (float): min-support value used for producing above results

The code also prints Tables 4,5,6 as displayed in the paper 
    Table 4: Triplet combination of diseases; output of the frequent itemset analysis 
    Table 5: Quartet combination of diseases; output of the frequent itemset anaysis 
    Table 6: Top 25 disease combinations that have 0 empirical prevalence in the MEPS data
---------------------------------------------------------------------------------------------------------------------------
"""

import pickle, itertools, sys, time, os.path, math, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from operator import itemgetter
from collections import OrderedDict

path_to_codebase = 'codebase/'
sys.path.insert(0, path_to_codebase)

from codebase.utils import clean_preproc_data_real
from codebase.extract_features import ExtractFeatures
from codebase.robust_optimizer import Optimizer as Optimizer_robust
from codebase.mba import marketbasket
from scipy.optimize import curve_fit

# Global variable 
dictionary_pc = {48:"Thyroid", 49:"Diabetes", 53:"Chol", 89:"Blind", 92:"Otitis", 95:'Nervous', 98:'HBP', 101:'Heart',
                126:'Upper Resp Infections', 127:'Lung', 128:'Asthma', 133:'Lower Resp', 134:'Upper Resp', 
                200:'Skin', 203:'Bone', 204:'Joint', 205:'Spondylosis', 211:'Tissue', 651:'Anxiety', 657:'Depression'}
dictionary_num = {0:"Lung", 1:"Heart", 2:"Lower Resp", 3:"Diabetes", 4:"Tissue", 5:"Joint", 6:"Anxiety", 7:"Skin", 8:"Blind",
                    9:"Depression", 10:"Upper Resp", 11:"Thyroid", 12:"Otitis", 13:"HBP", 14:"Spondylosis", 15:"Chol", 16:"Upper Resp Infections",
                    17:"Asthma", 18:"Bone", 19:"Nervous"}

# Function to read the probability distribution from the pickle file
def read_prob_dist(filename):
    with open(filename, "rb") as outfile:
        prob = pickle.load(outfile)                # If error, use encoding='latin1' as option while loading the pickle file
    return prob[1]

# Function to retreive the trained support predictor
def read_model(filename):
    with open(filename,"rb") as outfile:
        predictor = pickle.load(outfile)
    return predictor

def get_mle_prob(cleaneddata):
    """
    Find the empirical probabilities of all 2**k disease combinations
    Args:
        cleaneddata: the pandas Dataframe of all patient vectors
    Returns: 
        emp: probability distribution vector for all combinations 
    """
    #ignore warnings for pandas dataframe handling 
    pd.options.mode.chained_assignment = None  # default='warn'

    data = cleaneddata
    size = data.shape[0]
    diseases = data.shape[1]
    cols = np.arange(diseases)
    data.columns = cols

    # initialize mle
    zero_vectors = []
    mle = {}
    mle_sum = np.zeros(diseases+1)
    total_prob = 0.0

    ndata = data.groupby(list(data.columns)).size().to_frame('size').reset_index()
    ndata['size']/=size
    ndata['combi'] = ndata[list(data.columns)].values.tolist()
    ndata = ndata[['combi','size']]
    
    all_perms = itertools.product([0,1], repeat=diseases)
    for vec in all_perms:
        prob = ndata[ndata['combi'].apply(lambda x: x==list(vec))]['size'].values
        j = sum(vec)
        if prob.size==0:
            # mle.append(0)
            mle[vec] = 0
            zero_vectors.append(vec)
        else:
            # mle.append(prob[0])
            mle[vec] = prob[0]
            mle_sum[j] += prob[0]
            total_prob += prob[0]

    return zero_vectors, mle, mle_sum

# Function to compute maxent and empirical probabilities after optimization
def compute_prob_exact(optobj):
    """
    INPUT:
    -----
    optobj (Optimizer object) : Takes the optimizer object which has access to the parameters after optimization 

    OUTPUT: 
    -------
    vecprob (dict): Dictionary of probabilities obtained after running maxent on the MEPS data
    maxent_prob (list): List of probabilities obtained after running maxent on the MEPS data 
    maxent_sum_diseases (list): List of length (d+1) containing the disease-cardinalities from maximum entropy distribution
    emp_prob (list): List of length (d+1) containing the disease-cardinalities from MEPS data
    """
    vecprob = dict()
    maxent_prob = []
    num_feats = optobj.feats_obj.data_arr.shape[1]
    
    maxent_sum_diseases = np.zeros(num_feats+1)
    all_perms = itertools.product([0, 1], repeat=num_feats)
    total_prob = 0.0    # finally should be very close to 1

    for tmp in all_perms:
        vec = np.asarray(tmp)
        p_vec = optobj.prob_dist(vec)
        vecprob[tmp] = p_vec
        j = sum(vec)
        maxent_sum_diseases[j] += p_vec
        total_prob += p_vec
        maxent_prob.append(p_vec) 
    
    # print("Maxent probabilities: ", maxent_sum_diseases)
    # print('Total Probability: ', total_prob)
    # print()

    return vecprob, maxent_prob, maxent_sum_diseases

def get_data_char(cleaneddata):
    """
    Input: 
    ------
    cleaneddata 

    Output:
    -------
    d, N, pi_s 
    """
    # Step 1: Get d and N from the pandas dataframe 
    d = len(cleaneddata.columns)
    N = len(cleaneddata)

    def curve_fitting(cleaneddata, d, plot=False):    
        """
        Input:
        ------
        cleaneddata (pandas dataframe): Dataframe containing the input data 
        d (int): number of diseases 

        Output: 
        -------
        pi_s (float): Curve characteristics 
        """
        x = np.arange(0, d+1)
        freq = cleaneddata.sum(axis=1).value_counts().to_numpy() 
        y = np.pad(freq, (0,d-len(freq)+1), 'constant')
        y = y/np.sum(y)

        # Define a normalized probability distribution that I will do the curve fit for 
        def norm_exponential(x, pi_s):
            prob_dist = np.exp(-x/pi_s)    
            return prob_dist/prob_dist.sum()

        popt, pcov = curve_fit(norm_exponential, x, y)

        if plot == True:
            plt.figure()
            plt.plot(x, y, 'k*', label="Original Data")
            plt.plot(x, norm_exponential(x, *popt), 'r-', label="Fitted Curve")
            plt.xlabel("Number of diseases")
            plt.ylabel("Proportion of people having x number of diseases")
            plt.legend()
            plt.savefig('figures/meps_prevalence_fit.png')

        return popt[0]
    
    pi_s = curve_fitting(cleaneddata, d, True)
    return d, N, pi_s

def main(time_calc=True):
    """
    INPUT:
    ------
    None
    
    OUTPUT: 
    -------
    Stores the maximum entropy distribution for all 2^d vectors in the corresponding output file for MEPS
    """
    if time_calc == True: 
        sT = time.time() 

    # MEPS Data 
    meps_data_file = '../data/meps_data_prevalence.csv'
    
    # Sanity check to see if file exists (Just in case there was an error in generating the file)
    file_flag = os.path.isfile(meps_data_file)
    if file_flag == False:
        print("File does not exist, no MEPS data\n")
        return 

    # Step 1: Clean meps data (read from csv, and transform into the format used for processing)
    mapping, cleaneddata = clean_preproc_data_real(meps_data_file)

    # Step 2: Predict support using the support predictor 
    d, N, pi_s = get_data_char(cleaneddata)

    # Scale the features 
    scaled_d = 1
    scaled_N = 1
    scaled_pi_s = (pi_s - 1.0)/(1.8 - 1.0)

    # Retrieve random forest regressor model, and use it for predicting support
    predictor = read_model('output/support_predictor_2.pickle')
    covariates = np.array([scaled_d, scaled_N, scaled_pi_s]).reshape(1,-1)

    # Convert to polynomial features 
    from sklearn.preprocessing import PolynomialFeatures 
    poly = PolynomialFeatures(degree = 3) 
    support = predictor.predict(poly.fit_transform(covariates))[0]

    # Print the support predicted
    print("Features of the MEPS data and predicted support")
    print("d: ", d, " N: ", N, " pi_s: ", pi_s, " Support: ", support)

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
    # print("The clusters are:\n", feats.feat_partitions)

    # Calculate total number of constraints to be used in the maximum entropy computation
    num_constraints = len(two_wayc) + len(three_wayc) + len(four_wayc) 
    print("The number of frequent itemset constraints are: ", num_constraints)

    # Post process MEPS constraints and write them to a CSV file 
    # Inverse of mappings from key to value for MEPS codes
    inv_mappings = dict([[v,int(k)] for k,v in mapping.items()])

    pair_constraints = dict()
    for two in two_wayc:
        pair_constraints[tuple([dictionary_num[x] for x in two])] = round(support_dict[two],4)
    triple_constraints = dict()
    for three in three_wayc:
        triple_constraints[tuple([dictionary_num[x] for x in three])] = round(support_dict[three],4)
    quad_constraints = dict()
    for four in four_wayc:
        quad_constraints[tuple([dictionary_num[x] for x in four])] = round(support_dict[four],4)

    sorted_two = sorted(pair_constraints.items(), key=itemgetter(1), reverse=True)
    sorted_three = sorted(triple_constraints.items(), key=itemgetter(1), reverse=True)
    sorted_four = sorted(quad_constraints.items(), key=itemgetter(1), reverse=True)

    df1 = pd.DataFrame.from_records(sorted_two, columns=['Constraints','Support'])
    df2 = pd.DataFrame.from_records(sorted_three, columns=['Constraints','Support'])
    df3 = pd.DataFrame.from_records(sorted_four, columns=['Constraints','Support'])
    df = pd.concat([df1, df2, df3])

    # Write all constraints to CSV file 
    df.to_csv('../results/meps_prev_cons.csv', index=False)

    # Write support values to a csv file
    df = df.sort_values(by=['Support'], ascending=False)
    df.to_csv('../results/meps_prev_support.csv', index=False)


    # Step 6: Print the three way and four way constraints 
    def write_constraints_three():
        """
        Function to read in the three way constraints dictionary and output the corresponding combination pair words
        """
        three_way_converted = {}
        three_way_obs = {}
        for k, v in triple_constraints.items(): 
            three_way_converted[k] = math.floor(float(v)*320*(10**6))
            three_way_obs[k] = math.floor(float(v)*61166)
            
        df = pd.DataFrame(three_way_converted.items(), columns=['Disease Combinations','Prevalence in US Population'])
        df2 = pd.DataFrame(three_way_obs.items(), columns=['Disease Combinations','Number of observations in MEPS'])
        data = df2.set_index('Disease Combinations').join(df.set_index('Disease Combinations'))
        data = data.sort_values(by=['Number of observations in MEPS'], ascending=False)
        
        data['Prevalence in US Population'] = data.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
        data['Number of observations in MEPS'] = data.apply(lambda x: "{:,}".format(x['Number of observations in MEPS']), axis=1)
        print("\nTriplet disease constraints: ")
        print(data) 

    def write_constraints_four():
        """
        Function to read in the four way constraints dictionary and output the corresponding combination pair words
        """
        four_way_converted = {}
        four_way_obs = {}
        for k, v in quad_constraints.items(): 
            four_way_converted[k] = math.floor(float(v)*320*(10**6))
            four_way_obs[k] = math.floor(float(v)*61166)

        df = pd.DataFrame(four_way_converted.items(), columns=['Disease Combinations','Prevalence in US Population'])
        df2 = pd.DataFrame(four_way_obs.items(), columns=['Disease Combinations','Number of observations in MEPS'])
        data = df2.set_index('Disease Combinations').join(df.set_index('Disease Combinations'))
        data = data.sort_values(by=['Number of observations in MEPS'], ascending=False)
        
        data['Prevalence in US Population'] = data.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
        data['Number of observations in MEPS'] = data.apply(lambda x: "{:,}".format(x['Number of observations in MEPS']), axis=1)
        print("\nQuartet disease constraints: ")
        print(data) 

    write_constraints_three()
    write_constraints_four()

    # Step 7: Call on the L-BFGS-B optimizer to find the appropriate theta and run maximum entropy 
    opt = Optimizer_robust(feats, 1.0)
    soln_opt = opt.solver_optimize()
    if soln_opt == None:
        print('Solution does not converge')             # Error message when the optimizer does not reach convergence
        return 

    # Step 8: Calculate the maxEnt probabilities using the optimized parameters
    vecprob, maxent, sum_prob_maxent = compute_prob_exact(opt)
    print()

    # Step 9: Calculate the empirical probabilities 
    zeros, emp, sum_prob_emp = get_mle_prob(cleaneddata)

    # Step 10: For all zero vectors find the corresponding maxent probabilities
    pot_zeros = dict()
    for vector in zeros: 
        pot_zeros[vector] = vecprob[vector]

    sorted_vecprob = OrderedDict(sorted(pot_zeros.items(), key=itemgetter(1), reverse=True))

    # Store the zero vectors in a pickle file
    with open("output/meps_prev_zeroemp.pickle", 'wb') as outfile: 
        pickle.dump(sorted_vecprob, outfile)

    # Print the top 25 zero probability vectors
    top = list(sorted_vecprob.items())[:25]
    top_25_meps_code = {}

    # Write results to a text file in results
    with open('../results/maxent_mcc_zeroemp.txt', 'w') as f: 
        f.write("Top 25 vectors that have zero empirical probability\n")    
        for k, v in top:
            f.write('%s : %s \n' %([inv_mappings[x] for x in np.nonzero(k)[0]], v))
            top_25_meps_code[tuple([inv_mappings[x] for x in np.nonzero(k)[0]])] = v 

    def write_combinations():
        """
        Function to read in the top 25 most prevalent disease combinations and write their corresponding prevalence in a dataframe
        """
        top_25_converted = {}
        for k, v in top_25_meps_code.items():
            key_str = ""
            for c in k: 
                key_str += dictionary_pc[c] + ', '
            key_str = key_str.rstrip(', ')
            top_25_converted[key_str] = math.floor(float(v)*320*(10**6))

        df = pd.DataFrame(top_25_converted.items(), columns=['Disease Combinations', 'Prevalence in US Population'])

        df['Prevalence in US Population'] = df.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
        print("\nTop 25 most prevalent disease combinations having zero empirical prevalence")
        print(df.to_string(index=False)) 

    write_combinations()

    # Step 11: Store the probabilities in the corresponding output file
    outfilename = 'output/meps_prevalence_output.pickle'
    with open(outfilename, "wb") as outfile:
        pickle.dump((maxent, sum_prob_maxent, emp, sum_prob_emp, num_constraints, support), outfile)
    
    if time_calc==True:
        eT = time.time()
        print('Computational time for calculating maxent = {} seconds'.format(eT-sT))
    
    print()

if __name__ == '__main__':
    main()