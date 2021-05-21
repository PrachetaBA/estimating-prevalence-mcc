"""
Function/Code to convert ICD-9 codes to words for figures and tables in the paper
""" 
import pandas as pd 
import math
import re 

dictionary = {48:"THYROID", 49:"DIABETES", 53:"CHOL", 89:"BLIND", 92:"OTITIS", 95:'NERVOUS', 98:'HBP', 101:'HEART',
				126:'UPPER RESP INFECTIONS', 127:'LUNG', 128:'ASTHMA', 133:'LOWER RESP', 134:'UPPER RESP', 
				200:'SKIN', 203:'BONE', 204:'JOINT', 205:'SPONDYLOSIS', 211:'TISSUE', 651:'ANXIETY', 657:'DEPRESSION'}

dictionary_pc = {48:"Thyroid", 49:"Diabetes", 53:"Chol", 89:"Blind", 92:"Otitis", 95:'Nervous', 98:'HBP', 101:'Heart',
				126:'Upper Resp Infections', 127:'Lung', 128:'Asthma', 133:'Lower Resp', 134:'Upper Resp', 
				200:'Skin', 203:'Bone', 204:'Joint', 205:'Spondylosis', 211:'Tissue', 651:'Anxiety', 657:'Depression'}

three_way = {(49, 98, 53): 0.0249, (204, 98, 53): 0.0145, (101, 98, 53): 0.0098, (49, 204, 98): 0.0081, (98, 205, 53): 0.0075, (98, 53, 203): 0.0068, (49, 204, 53): 0.0067, (211, 98, 53): 0.0062, (657, 98, 53): 0.0056, (48, 98, 53): 0.0054, (101, 49, 98): 0.0054, (204, 98, 205): 0.0051, (127, 98, 53): 0.0049, (200, 98, 53): 0.0048, (211, 204, 98): 0.0045, (101, 49, 53): 0.0044, (98, 53, 126): 0.0041, (49, 98, 205): 0.004, (651, 98, 53): 0.0039, (211, 204, 205): 0.0038, (204, 205, 53): 0.0037, (98, 53, 95): 0.0037, (98, 53, 128): 0.0037, (651, 657, 98): 0.0036, (49, 211, 98): 0.0035, (49, 205, 53): 0.0035, (211, 204, 53): 0.0034, (49, 98, 203): 0.0034, (101, 204, 98): 0.0034, (651, 657, 205): 0.0031, (204, 651, 657): 0.0031, (204, 657, 98): 0.0031, (211, 98, 205): 0.003, (49, 657, 98): 0.003, (133, 98, 53): 0.0029}

four_way = {(49, 204, 98, 53):	0.0057,
(101, 49, 98, 53):	0.0039}

top_25 = {(204, 92, 98)  :  8.595978542680981e-05,
(92, 53, 126)  :  5.836985151096983e-05,
(200, 126, 95)  :  5.645152987953536e-05,
(49, 92, 53)  :  5.4140386340702485e-05,
(204, 200, 98, 205)  :  5.370987893857853e-05,
(211, 98, 205, 53, 203)  :  5.161412887270611e-05,
(49, 204, 98, 95)  :  4.8691429189821715e-05,
(49, 211, 48, 98, 53)  :  4.749916589453944e-05,
(101, 49, 126)  :  4.7105873219006635e-05,
(92, 98, 205)  :  4.424433450598904e-05,
(101, 204, 98, 205)  :  4.4082374339184144e-05,
(651, 98, 126)  :  4.407685440770868e-05,
(651, 200, 657, 205)  :  4.302170905443378e-05,
(211, 205, 128)  :  4.299305279041283e-05,
(49, 204, 98, 205, 53, 203)  :  4.2840698027718997e-05,
(651, 657, 92, 126)  :  4.2807866061816865e-05,
(49, 211, 98, 53, 95)  :  4.2101664233916055e-05,
(101, 92, 126)  :  4.1149407020266587e-05,
(204, 134, 92)  :  4.060576509422927e-05,
(133, 134, 98)  :  3.877556123069848e-05,
(200, 92, 205)  :  3.821805509082025e-05,
(98, 126, 95)  :  3.721966821569423e-05,
(127, 101, 126)  :  3.668264390410937e-05,
(657, 134, 205)  :  3.611581481335947e-05,
(127, 101, 49, 204, 98, 53)  :  3.5743082736319274e-05
}

def write_constraints_three():
	"""
	Function to read in the three way constraints dictionary and output the corresponding combination pair words
	"""
	three_way_converted = {}
	three_way_obs = {}
	for k, v in three_way.items(): 
		key_str = ""
		for c in k: 
			key_str += dictionary_pc[c] + ', '
		key_str = key_str.rstrip(', ')
		three_way_converted[key_str] = math.floor(float(v)*320*(10**6))
		three_way_obs[key_str] = math.floor(float(v)*61166)

	df = pd.DataFrame(three_way_converted.items(), columns=['Disease Combinations','Prevalence in US Population'])
	df2 = pd.DataFrame(three_way_obs.items(), columns=['Disease Combinations','Number of observations in MEPS'])
	data = df2.set_index('Disease Combinations').join(df.set_index('Disease Combinations'))
	data = data.sort_values(by=['Number of observations in MEPS'], ascending=False)
	
	data['Prevalence in US Population'] = data.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
	data['Number of observations in MEPS'] = data.apply(lambda x: "{:,}".format(x['Number of observations in MEPS']), axis=1)
	print(data) 

def write_constraints_four():
	"""
	Function to read in the four way constraints dictionary and output the corresponding combination pair words
	"""
	four_way_converted = {}
	four_way_obs = {}
	for k, v in four_way.items(): 
		key_str = ""
		for c in k: 
			key_str += dictionary_pc[c] + ', '
		key_str = key_str.rstrip(', ')
		four_way_converted[key_str] = math.floor(float(v)*320*(10**6))
		four_way_obs[key_str] = math.floor(float(v)*61166)

	df = pd.DataFrame(four_way_converted.items(), columns=['Disease Combinations','Prevalence in US Population'])
	df2 = pd.DataFrame(four_way_obs.items(), columns=['Disease Combinations','Number of observations in MEPS'])
	data = df2.set_index('Disease Combinations').join(df.set_index('Disease Combinations'))
	data = data.sort_values(by=['Number of observations in MEPS'], ascending=False)
	
	data['Prevalence in US Population'] = data.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
	data['Number of observations in MEPS'] = data.apply(lambda x: "{:,}".format(x['Number of observations in MEPS']), axis=1)
	print(data) 


def write_combinations():
	"""
	Function to read in the top 25 most prevalent disease combinations and write their corresponding prevalence in a dataframe
	"""
	top_25_converted = {}
	for k, v in top_25.items():
		key_str = ""
		for c in k: 
			key_str += dictionary_pc[c] + ', '
		key_str = key_str.rstrip(', ')
		top_25_converted[key_str] = math.floor(float(v)*320*(10**6))

	# print(top_25_converted)

	df = pd.DataFrame(top_25_converted.items(), columns=['Disease Combinations', 'Prevalence in US Population'])

	df['Prevalence in US Population'] = df.apply(lambda x: "{:,}".format(x['Prevalence in US Population']), axis=1)
	print(df.to_string(index=False)) 

def calc_combinations(): 
	"""
	Function to calculate the total population in the top 25 most prevalent disease combinations that are empirically zero. 
	"""
	total = 0
	for k, v in top_25.items(): 
		total += float(v)

	print("Population = ", total*320*10**6)

if __name__ == '__main__':
	# write_constraints_three()
	# write_constraints_four()
	# write_combinations()
	calc_combinations()