#!/usr/bin/env python3

"""
Created on Wed Mar 3 17:07:13 2021
@author: prachetaamaranath
"""

import numpy as np 
import pandas as pd 
import pickle, sys
import matplotlib.pyplot as plt 

# Machine Learning methods from scikit-learn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import warnings
warnings.filterwarnings("ignore")


def optimal_support(ds_n, div=1):
	"""
	INPUT: 
	------
	ds_n (int): Dataset Number
	div (int): 1 if JS distance, 2 if power divergence

	OUTPUT:
	-------
	
	"""
	
	# Step 1: Read the power divergence values for all synthetic data, preprocessing and cleaning up the data
	if div == 1: 
		supp_ds = pd.read_csv('../data/parameters/js_distance_'+str(ds_n)+'.csv')
	elif div == 2: 
		supp_ds = pd.read_csv('../data/parameters/power_divergence_'+str(ds_n)+'.csv')

	# Drop those rows with errors (either because of Power divergence calculation errors or timeout etc), or those 
	# that have -1 for the support and power_divergence
	supp_ds = supp_ds.dropna()
	supp_ds = supp_ds[supp_ds['div'] != -1]

	# Print the total number of datapoints available 
	print("Usable data points: ", supp_ds.shape[0])
	
	# Convert from object datatype to numeric
	supp_ds['div'] = pd.to_numeric(supp_ds['div'], errors='coerce')
	supp_ds['s'] = pd.to_numeric(supp_ds['s'], errors='coerce')
	supp_ds['c'] = pd.to_numeric(supp_ds['c'], errors='coerce')
	
	# Check the maximum and minimum power divergence values for reference 
	print("Maximum value: \n", supp_ds[supp_ds['div']==supp_ds['div'].max()]['div'])
	print("Minimum value: \n", supp_ds[supp_ds['div']==supp_ds['div'].min()]['div'])

	# Step 2: Choose the best power divergence values among all supports to create the machine learning input data 
	ml_data = supp_ds.sort_values("div").groupby(["f_n"], as_index=False).first()	# The first() option breaks ties by choosing the lowest/first available support value 

	# Step 3: Creating features and labels and split into training and test datasets
	scaler = MinMaxScaler() 
	features = ml_data.iloc[:, 2:5].values 
	labels = ml_data.iloc[:, 6].values 
	scaled_f = scaler.fit_transform(features)
	X_train, X_test, y_train, y_test = train_test_split(scaled_f, labels, test_size=0.15, random_state=5)

	# Define custom error function 
	def mean_abs_error(y_true, y_pred):
		mae = np.mean(np.abs(y_true - y_pred))
		return mae

	# Dictionary with all errors: 
	model_error = -np.inf
	best_model_text = "N/A"
	best_model = None

	# Test errors from different machine learning models
	# Model 1: Linear Regression
	ml_lr = LinearRegression()
	lin_reg_scores = cross_val_score(ml_lr, scaled_f, labels, cv=5, scoring='neg_mean_absolute_error') # scoring=make_scorer(mean_abs_error, greater_is_better=True))
	print("Linear Regression Value: ", np.mean(lin_reg_scores))

	# Check error and store best model each time 
	if np.mean(lin_reg_scores) > model_error: 
		model_error = np.mean(lin_reg_scores)
		best_model_text = "Linear Regression"
		best_model = LinearRegression().fit(scaled_f, labels)


	# Model 2: Polynomial Regression
	for degree in [2,3,4,5]:
		ml_pr = PolynomialFeatures(degree=degree)
		poly_scores = cross_val_score(ml_lr, ml_pr.fit_transform(scaled_f), labels, cv=5, scoring='neg_mean_absolute_error') # scoring=make_scorer(mean_abs_error, greater_is_better=True))
		print("Polynomial Regression Degree: ", degree, " Value: ", np.mean(poly_scores))
		# Check error and store best model each time 
		if np.mean(poly_scores) > model_error: 
			model_error = np.mean(poly_scores)
			best_model_text = "Polynomial Regression, Degree "+str(degree)
			poly = ml_pr.fit_transform(scaled_f)
			best_model = LinearRegression().fit(poly, labels)

	# Model 3: Random Forest regression 
	for depth in [1,2,3]:
		ml_rf = RandomForestRegressor(max_depth=depth, random_state=3, criterion='mae')
		rf_scores = cross_val_score(ml_rf, scaled_f, labels, cv=5, scoring='neg_mean_absolute_error') #scoring=make_scorer(mean_abs_error, greater_is_better=True))
		print("Random Forest regression Depth: ", depth, " Value: ", np.mean(rf_scores))

		# Check error and store best model each time 
		if np.mean(rf_scores) > model_error: 
			model_error = np.mean(rf_scores)
			best_model_text = "Random Forest Regression, Depth "+str(depth)
			best_model = ml_rf.fit(scaled_f, labels)
	

	# Model 4: Multi-Layer Perceptron
	for layer in [2]:
		ml_nn = MLPRegressor(hidden_layer_sizes=(layer,), random_state=3)
		nn_scores = cross_val_score(ml_nn, scaled_f, labels, cv=5, scoring='neg_mean_absolute_error') #scoring=make_scorer(mean_abs_error, greater_is_better=True))
		print("MLP Hidden Layers: ", layer, " Value: ", np.mean(nn_scores))

		# Check error and store best model each time 
		if np.mean(nn_scores) > model_error: 
			model_error = np.mean(nn_scores)
			best_model_text = "MLP, Hidden Layers "+str(layer)
			best_model = ml_nn.fit(scaled_f, labels)

	# Model 5: Decision Tree Regressor 
	for depth in [1,2,3]:
		ml_dt = DecisionTreeRegressor(max_depth=depth, random_state=6, criterion='mae')
		dt_scores = cross_val_score(ml_dt, scaled_f, labels, cv=5, scoring='neg_mean_absolute_error')
		print("DecisionTreeRegressor Depth: ", depth, ' Value: ', np.mean(dt_scores))

		# Check error and store best model each time 
		if np.mean(dt_scores) > model_error: 
			model_error = np.mean(dt_scores)
			best_model_text = "Decision Tree Regression, Depth "+str(depth)
			best_model = ml_dt.fit(scaled_f, labels)

	print()
	print("BEST MODEL: ", best_model_text)

	"""
	# Optinally, save model of our choosing as the best model (Polynomial Regression Degree 3)
	ml_pr = PolynomialFeatures(degree=3)
	poly = ml_pr.fit_transform(scaled_f)
	best_model = LinearRegression()
	best_model.fit(poly, labels)
	"""

	# Save the model after checking which is the best
	model_file = 'output/support_predictor_'+str(ds_n)+'.pickle'
	with open(model_file, 'wb') as f: 
		pickle.dump(best_model, f)


if __name__ == '__main__':
	ds_n = sys.argv[1]
	optimal_support(ds_n)