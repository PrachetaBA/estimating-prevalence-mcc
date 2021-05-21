# Estimating the Prevalence of Multiple Chronic Diseases via Maximum Entropy

Accompanying code for the paper ``Estimating the Prevalence of Multiple Chronic Diseases via Maximum Entropy``

To reproduce the procedure exactly as we have followed in the paper, we need to generate synthetic datasets ``(Delta_0, Delta_1, Delta_2)`` and run ``MaxEnt-MCC`` on the obtained synthetic datasets. We also used a compute cluster to parallelize the synthetic data generation and maxEnt algorithm processing. In this document, we first outline the procedure for running Algorithm 1 (``MaxEnt-MCC``) on the MEPS data and then proceed to describe the synthetic data generation process in detail. 

## Install software dependencies  
```
# For pip
python3 -m venv env
source env/bin/activate
pip install -r pip-requirements.txt

# For conda
conda create --name <env> --file requirements.txt
```

## Run ``MaxEnt-MCC``
```
cd src/
python meps_maxent.py
```
For a thorough description of the output files produced, see code documentation in the header of ``src/meps_maxent.py``

## Synthetic data experiments
Run all commands from the ``src/`` folder unless explicitly specified. 

### GenSynthDatasets
1. Generate synthetic data parameters: 
``python generate_params.py ds_n`` where ds_n represents the dataset number (**1,2,3**) corresponding to the three different datasets ``Delta_0, Delta_1, Delta_2`` we need to generate for futher experiments. 
2. Generate synthetic data: 
``python generate_data.py ds_n f_n`` where f_n is the file number of the corresponding parameter file, in our experiments this number extends from 1-2100 and ds_n is the dataset number corresponding to the three different datasets

### Learning optimal support 
1. Generate support data parameters: Run``python generate_support_params.py $ds_n`` where ds_n = 1
2. Run unregularized maxent using 10 different values of support on each of the files ``python synthetic_support_unreg.py $f_n $s_n`` where f_n specifies the file number ranging from 1-2100 and s_n is the support number ranging from 1-10)
3. Calculate the JS distance ``python extract_js.py $keyword`` 
where the keyword is ``support_1`` for the resulting maxent distributions. 
4. Extract the best performing features, create the training dataset and create the predictor for optimal support for unregularized maxEnt ``python learn_optimal_support.py $ds_n`` where ds_n = 1. 

### Learn best value of regularization parameters W 
1. For different values of width, run regularized maxent: ``python test_regularization.py $f_n $w_n`` (where f_n specifies the file number ranging from 1-2100 and w_n is the width number ranging from 0-15, 0 being unregularized maxent and 1-15 being different values for the width parameter W)
2. To visualize the JS distance for both MaxEnt and MLE 
```
cd output/plots
python plot_boxplots_reg.py
```

### Learn optimal support with regularization
1. Generate support data parameters: Run ``python generate_support_params.py 2``
2. Run regularized maxent using 10 different values of support on each of the files ``python synthetic_support_reg.py $f_n $s_n``(where f_n specifies the file number ranging from 1-2100 and s_n is the support number ranging from 1-10)
3. Calculate the JS distance ``python extract_js.py $keyword`` where the keyword is ``support_2``. 
4. Extract the best performing features, create the training dataset and create the predictor for optimal support for unregularized maxEnt: ``python learn_optimal_support.py $ds_n`` where ds_n = 2. 

### Comparing MaxEnt estimator with MLE estimator.
1. For synthetic dataset ``Delta_2`` (ds_n = 3), we can compare the performance of MaxEnt-MCC with the MLE estimator: ``python maxent_mle.py $f_n`` where f_n ranges from 1-2100. 
2. Calculate the JS distance ``python extract_js.py $keyword`` where the keyword is ``mle``. 
3. To visualize the JS distance for both MaxEnt and MLE 
```
cd output/plots
python plot_boxplots_mle.py
```
