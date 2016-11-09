import pandas as pd
import numpy as np
import pdb

def prepare_DP(states):
	"""
	Calculates an array with discrete values for the DP state
	and an index array
	
	Args:
		xstpes: number of discretisation steps
		xmin: Minimum value of DP state
		xmax: Maximum value of DP state
		
	Returns:
		Xidx: index array [1..xsteps]
		Xval: array with discrete values [xmin .. xmax]
		
	>>> prepare_DP(1,5,9)
	(array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
	array([ 1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ]))
	"""
	states['xmin'],states['xmax'],states['xsteps']
	
	Xidx = np.arange(states['xsteps'])
	dx = (states['xmax']-states['xmin'])/np.float(states['xsteps']-1)
	Xval = np.linspace(states['xmin'],states['xmax'],states['xsteps'])
	return Xidx,Xval,dx



def DP_forward(states,U,cst,srs,system,J0=None,Data0=None,verbose=False):
	"""
	Implementation of the Dynamic Programming Shortest Path Algorithm - forward in time implementation 
	
	Args:
		states: pandas DataFrame or dict containing
			xstpes:	number of discretisation steps
			xmin: 	Minimum value of DP state
			xmax: 	Maximum value of DP state
		U: 		List of possible decision
		cst: 	Constant Parameters for system simulation
			t_start: First timestep used for optimization
			t_end: 	Last timestep used for optimization
		srs: 	Time-series for system simulation
		system:	function to simulate the sistem that is optimized usin DP
		J0:		Cost vector for first time-step
		Data0: 	Already known Data for previous time-steps
		verbose: Bool, turn shell output on (True) or off (False)
		
	Returns:
		pii:	pandas DataFrame with optimal decisions for all timesteps (=columns),
				index represents end state 
		J: 		Vector with total costs, index represents end state 
		X: 		pandas DataFrame with states for all timesteps (=columns), index represents end state 
		Data:	pandas DataFrame with output Data form system for all timesteps (=columns),
				index represents end state
	"""
	#################
	#Prepare DP Data
	#################
	
	# Prepare parameters needed for DP
	# Xidx = vector containing indices of descretized States, starting from 0
	# Xval = vector containing descretized States
	# dx = step size of discretization
	Xidx,Xval,dx = prepare_DP(states)
	l = len(Xidx) #number of discrete steps of state 
	
	#Time
	t0 = cst['t_start']
	T = cst['t_end']
	timesteps = np.arange(t0,T+1)
	
	
	############
	#Initialize
	############
	if J0 is None:
		J = np.zeros(l)
	else:
		J = J0	
	Data=None
	
	####################
	#First Timestep t0
	####################
	#for all decisions u
	for u in U:
		cost_u,Xval_j_u,data_u = system(u,Xval,t0,cst,srs,Data) #simulate system for all states Xval
		Xidx_j_u = find_nearest(Xval_j_u,Xval,l) #find index of nearest disrete state
		if u == U[0]:
		#init Dataframes after first run of system
			idx_data_ = pd.MultiIndex.from_product([Xidx,U],names=['Xi','U'])
			columns_data_ = data_u.columns.union(['J','U','Xi','Xidx','Xidxj'])
			data_ = pd.DataFrame(index=idx_data_,columns = columns_data_ )
			Data = pd.DataFrame(index=pd.MultiIndex.from_product([np.append(timesteps,T+1),Xidx],names=['t', 'Xidx_end']),columns = data_.columns)
		mi = pd.MultiIndex.from_arrays([Xidx,[u]*l])
		data_.loc[mi,data_u.columns] = data_u.values
		data_.loc[mi,['J','Xi','Xidx','Xidxj']] = np.array([cost_u + J,Xval,Xidx,Xidx_j_u]).transpose()
		data_.loc[mi,'U'] = u

		
	data_.set_index('Xidxj',append=True,inplace=True,drop=False)
	idx = data_['J'].groupby(level='Xidxj').idxmin()
	data = pd.DataFrame(index=Xidx,columns = data_.columns).fillna(9999999999.9)
	data.loc[idx.index] = data_.loc[idx.values].values #select data for min costs
	J = data['J'].values #minimal costs for next state Xj
	Data.loc[t0] = data.values #add data to Dataframe for t0

	
	####################
	#Timestep t1 - end
	####################
	for t in timesteps[1:]:
		data_ = pd.DataFrame(index=idx_data_,columns = columns_data_ )
		if verbose:
			print('Timestep: ',t) #shell output
		
		#for all decisions u	
		for u in U:
			cost_u,Xval_j_u,data_u = system(u,Xval,t,cst,srs,Data) #simulate system for all states Xval
			Xidx_j_u = find_nearest(Xval_j_u,Xval,l) #find index of nearest disrete state
			mi = pd.MultiIndex.from_arrays([Xidx,[u]*l])
			data_.loc[mi,data_u.columns] = data_u.values
			data_.loc[mi,['J','Xi','Xidx','Xidxj']] = np.array([cost_u + J,Xval,Xidx,Xidx_j_u]).transpose()
			data_.loc[mi,'U'] = u

		data_.set_index('Xidxj',append=True,inplace=True,drop=False)
		idx = data_['J'].groupby(level='Xidxj').idxmin()
		data = pd.DataFrame(index=Xidx,columns = data_.columns).fillna(9999999999.9)
		data.loc[idx.index] = data_.loc[idx.values].values #select data for min costs
		J = data['J'].values #minimal costs for next state Xj
		Data.loc[t] = data.values #add data to Dataframe for t0
		X_i = data['Xidx'].values
		mi0 = pd.MultiIndex.from_product([range(t0,t),Xidx])
		mi1 = pd.MultiIndex.from_product([range(t0,t),X_i])
		Data.loc[mi0] = Data.loc[mi1].values
	Data['X'] = Data['Xi']
	Data.loc[T+1,'X'] = Xval
	Data.drop(['Xi','Xidx','Xidxj'],axis=1,inplace=True)
	return Data

def DP_backward(states,U,cst,srs,system,JT=None,verbose=False):
	"""
	Implementation of the Dynamic Programming Shortest Path Algorithm - backward in time implementation 
	
	Args:
		states: pandas DataFrame or dict containing
			xstpes:	number of discretisation steps
			xmin: 	Minimum value of DP state
			xmax: 	Maximum value of DP state
		U: 		List of possible decision
		cst: 	Constant Parameters for system simulation
			t_start: First timestep used for optimization
			t_end: 	Last timestep used for optimization
		srs: 	Time-series for system simulation
		system:	function to simulate the sistem that is optimized usin DP
		JT:		Cost vector for last time-step
		verbose: Bool, turn shell output on (True) or off (False)
		
	Returns:
		Data:	pandas DataFrame with output Data from system for all timesteps (index0),
				and end states (index1), contains all system outputs and
			U:	optimal decisions 
			J: 	total costs
			X: 	States at beginning of times-step t
	"""
	
	#################
	#Prepare DP Data
	#################
	
	# Prepare parameters needed for DP
	# Xidx = vector containing indices of descretized States, starting from 0
	# Xval = vector containing descretized States
	# dx = step size of discretization
	Xidx,Xval,dx = prepare_DP(states)
	l = len(Xidx) #number of discrete steps of state 
	
	#Time
	t0 = cst['t_start']
	T = cst['t_end']
	timesteps = np.arange(t0,T+1)
	
	############
	#Initialize
	############
	Data=None
	if JT is None:
		J = np.zeros(l)
	else:
		J = JT
	

	####################
	#Last Timestep t
	####################
	if verbose:
		print('Timestep: ',T) #shell output
	#for all decisions u	
	for u in U:
		cost_u,Xval_j_u,data_u = system(u,Xval,T,cst,srs,Data)#simulate system for all states Xval
		Xidx_j_u = find_nearest(Xval_j_u,Xval,l)#find index of nearest disrete state
		if u == U[0]:
			data_ = pd.DataFrame(index=pd.MultiIndex.from_product([Xidx,U],names=['Xi','U']),columns = data_u.columns.union(['J','U','Xj','Xidxj']))
			Data = pd.DataFrame(index=pd.MultiIndex.from_product([np.append(timesteps,T+1),Xidx],names=['t', 'Xidx_start']),columns = data_.columns)
		mi = pd.MultiIndex.from_arrays([Xidx,[u]*l])
		data_.loc[mi,data_u.columns] = data_u.values
		data_.loc[mi,['J','Xj','Xidxj']] = np.array([cost_u + J[Xidx_j_u],Xval[Xidx_j_u],Xidx_j_u]).transpose()
		data_.loc[mi,'U'] = u
			
	idx = data_['J'].groupby(level=0).idxmin() #index of min costs
	data = data_.loc[idx] #select data for min costs
	J = data['J'].values #minimal costs for previous state Xi
	Data.loc[T] = data.values #add data to Dataframe for T

	####################
	#Timesteps T-1 to 1
	####################

	for t in reversed(timesteps[:-1]):
		if verbose:
			print('Timestep: ',t) #shell output
			
		#for all decisions u		
		for u in U:
			cost_u,Xval_j_u,data_u = system(u,Xval,t,cst,srs,Data) #simulate system for all states Xval
			Xidx_j_u = find_nearest(Xval_j_u,Xval,l) #find index of nearest disrete state
			mi = pd.MultiIndex.from_arrays([Xidx,[u]*l])
			data_.loc[mi,data_u.columns] = data_u.values
			data_.loc[mi,['J','Xj','Xidxj']] = np.array([cost_u + J[Xidx_j_u],Xval[Xidx_j_u],Xidx_j_u]).transpose()
			data_.loc[mi,'U'] = u
				
		idx = data_['J'].groupby(level=0).idxmin() #index of min costs
		data = data_.loc[idx] #select data for min costs
		J = data['J'].values #minimal costs for previous state Xi
		Data.loc[t] = data.values #add data to Dataframe for T
		Xidx_j = data['Xidxj'].values
		mi0 = pd.MultiIndex.from_product([range(t+1,T+1),Xidx])
		mi1 = pd.MultiIndex.from_product([range(t+1,T+1),Xidx_j])
		Data.loc[mi0] = Data.loc[mi1].values		
		
	mit0 = pd.MultiIndex.from_product([range(t0+1,T+2),Xidx])
	mit1 = pd.MultiIndex.from_product([range(t0,T+1),Xidx])
	Data.loc[mit0,'X'] = Data.loc[mit1,'Xj'].values
	Data.loc[timesteps[0],'X'] = Xval#First State
	Data.drop(['Xj','Xidxj'],axis=1,inplace=True)
	return Data
	
	
def find_nearest(xj,Xval,l):
	"""
	Finds the index imin for each element of the array xj with respect to the discretized array Xval
	
	Args:
		xj: array of undiscrete values
		Xval: array of discrete values
		l: number of discrete steps
		
	Returns:
		imin: array that contains the index of the vector Xval which value is the nearest to xj
	"""
	# one=np.ones((l,l))
	# diff = np.abs(one*xj - np.transpose(one*Xval))
	# i_min = diff.argmin(axis=0)
	i_min = Xval.searchsorted(xj) #finds the index of Xval which is the closest GREATER value to xj 
	i_min = np.clip(i_min, 1, l-1)# making 0->1 and l->l-1; (cutting borders)
	left = Xval[i_min-1] #Value of Xval that is closest LOWER to xj (left border)
	right = Xval[i_min] #Value of Xval that is closest GREATER to xj (right border)
	i_min -= xj - left < right - xj #choose which border is closer and get index of Xval
	return i_min

