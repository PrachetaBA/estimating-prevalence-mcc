# Estimating the Prevalence of Multiple Chronic Diseases via Maximum Entropy

Accompanying code for the paper ``Estimating the Prevalence of Multiple Chronic Diseases via Maximum Entropy``

We outline the steps required to reproduce all experiments listed in the paper. The first step is to generate synthetic data ``(Datasets: Delta_0, Delta_1, Delta_2)`` and run experiments on them to parameterize and evaluate ``MaxEnt-MCC`` on the obtained (synthetic) datasets. 
Note: We have used the Swarm2 (a high performance compute) cluster to parallelize the synthetic data generation procedure. Each compute node in the cluster is a Xeon E5-2680 v4 @ 2.40GHz, has 128GB RAM and 200GB local SSD disk. While the README file outlines the procedure to run experiments sequentially, users can run all synthetic generation experiments in parallel. 

## Install software dependencies  
```
# For pip
python3 -m venv env
source env/bin/activate
pip install -r pip-requirements.txt

# For conda
conda create --name <env> --file requirements.txt
```

## Synthetic data experiments
This codebase uses relative paths. Run all commands from the ``code/`` folder unless explicitly specified. 

### GenSynthDatasets
1. Generate parameters for synthetic data (The parameters for the synthetic data are described in Table 2 of the paper): 
``python generate_params.py ds_n`` where ds_n represents the dataset number (**1,2,3**) corresponding to the three different datasets ``Delta_0, Delta_1, Delta_2`` we need to generate for futher experiments. 
2. Generate synthetic data: 
``python generate_data.py ds_n f_n`` where f_n is the file number of the corresponding parameter file, in our experiments this number extends from 1-2100 and ds_n is the dataset number corresponding to the three different datasets.

### Learn a model to predict optimal min-support 
#### Generate training data for min-support model 
1. Generate parameters for min-support training data: Run ``python generate_support_params.py 1`` (The first dataset ``Delta_0`` is used as training data for the min-support model with unregularized maxEnt)

#### Use training data to learn a model that predicts optimal min-support
1. Run unregularized maxEnt using 10 different values of support for each datum ``python synthetic_support_unreg.py $f_n $s_n`` where f_n specifies the file number ranging from 1-2100 and s_n is the support number ranging from 1-10)
2. Calculate the Jensen-Shannon distance between the ground-truth distribution and the resulting maxEnt distribution for each datum ``python extract_js.py support_1`` 
3. To learn a model that predicts the optimal min-support, for each file number ``$f_n``, we extract the min-support that results in the lowest Jensen-Shannon distance to the ground-truth distribution. The parameters(dataset size, number of diseases, exponential curve-fit parameter) of these datasets can now serve as features to train many machine-learning (ML) models. In our code, we select a few algorithms (linear regression, polynomial regression, multi-layer perceptrons, random forests and decision trees) to learn a model. Using cross-validation, we are able to find the model which has the highest prediction accuracy (lowest error). This code automatically chooses the model with the highest accuracy as the best predictor: ``python learn_optimal_support.py 1``  

### Learn optimal value of regularization parameters W 
1. For a range of values for the width parameter W, run regularized maxEnt on the training data: ``python test_regularization.py $f_n $w_n`` (where f_n specifies the file number ranging from 1-2100 and w_n is the width number ranging from 0-15, 0 being unregularized maxent and 1-15 being different values for the width parameter W)
2. To visualize the Jensen-Shannon distance between the resulting maxEnt distribution and the maximum-likelihood estimates, run: 
```
cd code/output/plots
python plot_boxplots_reg.py
```
These plots were used to analyze and obtain the optimal value of ``W`` used in the rest of the experiments.

### Create final machine-learning model for choosing min-support 
#### Generate training data for min-support model 
1. Generate parameters for min-support training data: Run ``python generate_support_params.py 2`` (The second dataset ``Delta_1`` is used as training data for the min-support model with regularized maxEnt)

#### Use training data to learn a model that predicts optimal min-support
1. Run regularized maxent using 10 different values of support for each datum ``python synthetic_support_reg.py $f_n $s_n``(where f_n specifies the file number ranging from 1-2100 and s_n is the support number ranging from 1-10)
2. Calculate the Jensen-Shannon distance between the ground-truth distribution and the resulting maxEnt distribution for each datum ``python extract_js.py support_2`` 
3. To learn a model that predicts the optimal min-support, for each file number ``$f_n``, we extract the min-support that results in the lowest Jensen-Shannon distance to the ground-truth distribution. The parameters(dataset size, number of diseases, exponential curve-fit parameter) of these datasets can now serve as features to train many machine-learning (ML) models. In our code, we select a few algorithms (linear regression, polynomial regression, multi-layer perceptrons, random forests and decision trees) to learn a model. Using cross-validation, we are able to find the model which has the highest prediction accuracy (lowest error). This code prints the output as seen in Table 1 of the Online Supplement and automatically chooses the model with the highest accuracy as the best predictor: ``python learn_optimal_support.py 2``  
Note: Alternatively, users can train a ML model of their choice on the constructed training dataset and choose to use that model for further experiments.

### Comparing MaxEnt estimator with maximum-likelihood estimator.
1. For synthetic dataset ``Delta_2`` (ds_n = 3), we can compare the performance of MaxEnt-MCC with the MLE estimator: ``python maxent_mle.py $f_n`` where f_n ranges from 1-2100. 
2. Calculate the Jensen-Shannon distance between the maxEnt/maximum-likehood distribution and the ground-truth distribution ``python extract_js.py mle``.
3. To visualize the Jensen-Shannon distance(Reproduces the Figure 2 of the paper):
```
cd code/output/plots
python plot_boxplots_mle.py
```

## Run ``MaxEnt-MCC`` on MEPS 
To run the ``MaxEnt-MCC`` algorithm on the MEPS data: 
```
cd code/
python meps_maxent.py
```
To ease the computational effort to reproduce synthetic data, in this codebase, we have included the final machine-learning model used for predicting optimal min-support. 
For a thorough description of the output files produced, see code documentation in the header of ``code/meps_maxent.py``