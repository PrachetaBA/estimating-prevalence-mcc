#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: prasannasrinivasan
@author: prachetaamaranath
"""
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from functools import reduce


def marketbasket(cleaneddata, support):
    '''
    Function to use market basket analysis to return 2 way, 3 way and 4 way constraints. 
    Input: 
        Data vectors from synthetically generated data 
        Support: Support for the fpgrowth algorithm
    Returns: 
        Dictionary containing 2 way, 3 way and 4 way constraints 
    '''
    frequent_itemsets = fpgrowth(cleaneddata, min_support=support, use_colnames=True)
    
    # Get the support of the marginals which satisfy the minimum support 
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(lambda x: len(x))
    rules1 = frequent_itemsets[(frequent_itemsets['length'] == 1)]['itemsets'].values
    m1 = [list(x) for x in rules1]
    if m1 == []:
        return {}, {}, {}, {}, {}
    m2 = reduce(lambda x,y: x+y, m1)
    marginals = [int(x) for x in m2]
    marginalsdict = dict.fromkeys(marginals, (1))

    try:
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=support)
        rules["itemsets"]=rules.apply(lambda x: x["antecedents"].union(x["consequents"]), axis=1)
        rules["itemsets"]=rules["itemsets"].apply(lambda x: [int(y) for y in x])
        rules["itemsets"]=rules["itemsets"].apply(lambda x: tuple(sorted(x)))
        rules["itemsets_len"]=rules["itemsets"].apply(lambda x: len(x))
        rules=rules.drop_duplicates(subset="itemsets")
        rules=rules.drop(['antecedents', 'consequents', 'antecedent support', 'consequent support', 'confidence', 'lift', 'leverage', 'conviction'], axis=1)
        rules2=rules["itemsets"][rules['itemsets_len']==2]
        rules3=rules["itemsets"][rules['itemsets_len']==3]
        rules4=rules["itemsets"][rules['itemsets_len']==4]
        supportdict=dict(zip(rules["itemsets"], rules["support"]))
        twowaydict=dict.fromkeys(rules2, (1, 1))
        threewaydict=dict.fromkeys(rules3, (1, 1, 1))
        fourwaydict=dict.fromkeys(rules4, (1, 1, 1, 1))
        
        return supportdict, marginalsdict, twowaydict, threewaydict, fourwaydict
    except:
        return {}, {}, {}, {}, {}