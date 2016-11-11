import pandas as pd
import numpy as np
import pdb

def prepare_DP(states):
	"""
	Create arrays with discrete values for each DP state
	
	Args:
		states: pandas dataframe where each index represents a state variable
			xmin: minimum value of the state variable
			xmax: maximum value of the state variable
			xstpes: number of discretization steps for this state variable
			
	Returns:
		Xi_val: 2d array containing every state combination of the state variables
		Xidx: array, which contains indices of discrete values Xi_val
		XX:	2d array, containing all discretized state variable arrays
			(Xi_val is the cartesian product of XX)
		xsteps: number of discretization steps for each state variable
		columns: columns used in the DataFrames within the DP algorithm (without U)
		columns_u: columns plus 'U'
	"""
	
	#reading data for building basic arrays
	num_states = len(states.index)
	xmin=states['xmin'].values
	xmax=states['xmax'].values
	xsteps=states['xsteps'].values

	#building discretized array for each state variable and save them to XX
	XX = []
	for i in range(num_states):
		X=np.linspace(xmin[i],xmax[i],xsteps[i])
		XX.append(X)
		

	#calculating Xi_val, containing every state combination of the state variables
	#(cartesian product of XX)
	Xi_val = np.array(pd.tools.util.cartesian_product(XX))
	if num_states == 1:
		Xi_val = Xi_val[0]
	
	#Index array for Xi_val
	Xidx = np.arange(xsteps.prod())

	# create list with column names for the DataFrames
	columns = ['J'] # insert J = total costs
	for i in range(num_states):
		columns = columns + ['Xi_val_'+states.index[i]] # For each state variable i, add a "Xi_val_statename" column
	columns = columns + ['Xidx','Xidxj'] #add columns "Xidx" and "Xidxj"
	for i in range(num_states):
		columns = columns + ['Xj_val_'+states.index[i]] # For each state variable i, add a "Xj_val_statename" column
	
	#Add column "U" for column_u
	columns_u = list(columns)
	columns_u.insert(1,"U")
	
   
	return Xi_val, Xidx, XX, xsteps, columns, columns_u



def DP_forward(states,U,timesteps,cst,srs,system,J0=None,verbose=False,t_verbose=1):
	"""
	Implementation of the Dynamic Programming Shortest Path Algorithm - forward in time implementation 
	
	Args:
		states: pandas dataframe where each index represents a state variable
			xmin: minimum value of the state variable
			xmax: maximum value of the state variable
			xstpes: number of discretization steps for this state variable
		U: 		List of possible decision
		timesteps: numpy array containing all timesteps
		cst: 	Constant Parameters for system simulation
		srs: 	Time-series for system simulation
		system:	function to simulate the system that is optimized using DP
		J0:		Cost vector for first time-step
		verbose: Bool, turn shell output on (True) or off (False)
		
	Returns:
		Data:	pandas DataFrame with results
			Data has to indices, where
			't': timestep
			'X_end': state at the end of the last timestep
	"""
	#################
	#Prepare DP Data
	#################
	
	# Prepare parameters needed for DP
	Xi_val, Xidx, XX, xsteps, columns, columns_u = prepare_DP(states)
	lenX = len(Xidx)
	
	#Time
	t0=timesteps[0]
	T=timesteps[-1]
	# timesteps = np.arange(t0,T+1)
	
	
	############
	#Initialize
	############
	if J0 is None:
		J = np.zeros(lenX )
	else:
		J = J0	
	Data=None
	
	#creating multiindex for each u in U
	MI = {}
	for u in U:
		MI[u] = pd.MultiIndex.from_arrays([Xidx,[u]*lenX ])
	
    #First Timestep t0
    ####################
    #for all decisions u
	if verbose:
		print('Timestep: ',t0) #shell output
	for u in U:
		cost_u,Xj_val_u,data_u = system(u,Xi_val,t0,cst,srs,Data) #simulate system for all states Xi_val
		Xj_val_u=np.atleast_2d(Xj_val_u) #force Xj_val_u to be at least 2d array (works in case of 1 state model)         
		Xidx_j_u = find_nearest(Xj_val_u,XX,xsteps) #find index of nearest disrete state
		if u == U[0]:
		#init two Dataframes after first run of system
			idx_data_ = pd.MultiIndex.from_product([Xidx,U],names=['Xi','U'])
			columns_data_ = data_u.columns.union(columns_u)
			data_ = pd.DataFrame(index=idx_data_,columns = columns_data_ )
			Data = pd.DataFrame(index=pd.MultiIndex.from_product([np.append(timesteps,T+1),Xidx],names=['t', 'Xidx_end']),columns = data_.columns)
		
		#filling first Dataframe
		mi = MI[u]
		data_.loc[mi,data_u.columns] = data_u.values
		data_.loc[mi,columns] = np.vstack([cost_u + J,Xi_val,Xidx,Xidx_j_u,Xj_val_u]).transpose()
		data_.loc[mi,'U'] = u

	#choosing decision for every index with minimal costs and saving it to second Dataframe
	data_.set_index('Xidxj',append=True,inplace=True,drop=False)
	idx = data_['J'].groupby(level='Xidxj').idxmin()
	data = pd.DataFrame(index=pd.Index(Xidx,name='Xidx'),columns = data_.columns).fillna(9999999999.9)
	data.loc[idx.index] = data_.loc[idx.values].values #select data for min costs
	J = data['J'].values #minimal costs for next state Xj
	Data.loc[t0] = data.values #add data to Dataframe for t0


	#####################
	#Timestep t1 - end
	#####################
	for t in timesteps[1:]:
		data_ = pd.DataFrame(index=idx_data_,columns = columns_data_ )
		if verbose and t%t_verbose==0:
			print('Timestep: ',t) #shell output
				
		#for all decisions u	
		for u in U:
			cost_u, Xj_val_u,data_u=system(u,Xi_val,t,cst,srs,Data) #simulate system for all states Xi_val
			Xj_val_u=np.atleast_2d(Xj_val_u) #force Xj_val_u to be at least 2d array (works in case of 1 state model)             
			Xidx_j_u = find_nearest(Xj_val_u,XX,xsteps)#find index of nearest disrete state
			mi = MI[u]
			data_.loc[mi,data_u.columns] = data_u.values
			data_.loc[mi,columns] = np.vstack([cost_u + J,Xi_val,Xidx,Xidx_j_u,Xj_val_u]).transpose()
			data_.loc[mi,'U'] = u

		data_.set_index('Xidxj',append=True,inplace=True,drop=False)
		idx = data_['J'].groupby(level='Xidxj').idxmin()
		data = pd.DataFrame(index=Xidx,columns = data_.columns).fillna(9999999999.9)
		data.loc[idx.index] = data_.loc[idx.values].values #select data for min costs
		J = data['J'].values #minimal costs for next state Xj
		Data.loc[t] = data.values #add data to Dataframe for t0
		
		#rewriting the existed path to the cheapest one        
		X_i = data['Xidx'].values
		mi0 = pd.MultiIndex.from_product([range(t0,t),Xidx])
		mi1 = pd.MultiIndex.from_product([range(t0,t),X_i])
		Data.loc[mi0] = Data.loc[mi1].values
		
	Data = modify_results(Data,states)	
	# Data['X'] = Data['Xi']
	# Data.loc[T+1,'X'] = Xval
	# Data.drop(['Xi','Xidx','Xidxj'],axis=1,inplace=True)
	return Data

def DP_backward(states,U,timesteps,cst,srs,system,JT=None,verbose=False,t_verbose=1):
	"""
	Implementation of the Dynamic Programming Shortest Path Algorithm - forward in time implementation 
	
	Args:
		states: pandas dataframe where each index represents a state variable
			xmin: minimum value of the state variable
			xmax: maximum value of the state variable
			xstpes: number of discretization steps for this state variable
		U: 		List of possible decision
		timesteps: numpy array containing all timesteps
		cst: 	Constant Parameters for system simulation
		srs: 	Time-series for system simulation
		system:	function to simulate the system that is optimized using DP
		J0:		Cost vector for first time-step
		verbose: Bool, turn shell output on (True) or off (False)
		
	Returns:
		Data:	pandas DataFrame with results
			Data has to indices, where
			't': timestep
			'X_start': state at the beginning of the first timestep
	"""
	
	#################
	#Prepare DP Data
	#################
	
	# Prepare parameters needed for DP
	Xi_val, Xidx, XX, xsteps, columns, columns_u = prepare_DP(states)
	Xi_val_2d = np.atleast_2d(Xi_val) #for one state, a 2d array of Xi_val is needed to implement a general model (used for indexing when creating data_)
	lenX = len(Xidx)
	
	#Time
	t0=timesteps[0]
	T=timesteps[-1]
	# timesteps = np.arange(t0,T+1)
	
	
	############
	#Initialize
	############
	if JT is None:
		J = np.zeros(lenX )
	else:
		J = JT	
	Data=None
	
	#creating multiindex for each u in U
	MI = {}
	for u in U:
		MI[u] = pd.MultiIndex.from_arrays([Xidx,[u]*lenX ])
		
		
	####################
	#Last Timestep t
	####################
	if verbose:
		print('Timestep: ',T) #shell output
	#for all decisions u	
	for u in U:
		cost_u,Xj_val_u,data_u = system(u,Xi_val,T,cst,srs,Data) #simulate system for all states Xi_val
		Xj_val_u=np.atleast_2d(Xj_val_u) #force Xj_val_u to be at least 2d array (works in case of 1 state model)         
		Xidx_j_u = find_nearest(Xj_val_u,XX,xsteps) #find index of nearest disrete state
		if u == U[0]:
		#init two Dataframes after first run of system
			idx_data_ = pd.MultiIndex.from_product([Xidx,U],names=['Xi','U'])
			columns_data_ = data_u.columns.union(columns_u)
			data_ = pd.DataFrame(index=idx_data_,columns = columns_data_ )
			Data = pd.DataFrame(index=pd.MultiIndex.from_product([np.append(timesteps,T+1),Xidx],names=['t', 'Xidx_start']),columns = data_.columns)
			
		#filling first Dataframe
		mi = MI[u]
		data_.loc[mi,data_u.columns] = data_u.values
		data_.loc[mi,columns] = np.vstack([cost_u + J[Xidx_j_u],Xi_val,Xidx,Xidx_j_u,Xi_val_2d[:,Xidx_j_u]]).transpose()
		data_.loc[mi,'U'] = u
	
	idx = data_['J'].groupby(level=0).idxmin() #index of min costs
	data = data_.loc[idx] #select data for min costs
	J = data['J'].values #minimal costs for previous state Xi
	Data.loc[T] = data.values #add data to Dataframe for T

	####################
	#Timesteps T-1 to 1
	####################

	for t in reversed(timesteps[:-1]):
		if verbose and t%t_verbose==0:
			print('Timestep: ',t) #shell output
			
		#for all decisions u		
		for u in U:
			cost_u, Xj_val_u,data_u=system(u,Xi_val,t,cst,srs,Data) #simulate system for all states Xi_val
			Xj_val_u=np.atleast_2d(Xj_val_u) #force Xj_val_u to be at least 2d array (works in case of 1 state model)             
			Xidx_j_u = find_nearest(Xj_val_u,XX,xsteps)#find index of nearest disrete state
			mi = MI[u]
			data_.loc[mi,data_u.columns] = data_u.values
			data_.loc[mi,columns] = np.vstack([cost_u + J[Xidx_j_u],Xi_val,Xidx,Xidx_j_u,Xi_val_2d[:,Xidx_j_u]]).transpose()
			data_.loc[mi,'U'] = u
				
		idx = data_['J'].groupby(level=0).idxmin() #index of min costs
		data = data_.loc[idx] #select data for min costs
		J = data['J'].values #minimal costs for previous state Xi
		Data.loc[t] = data.values #add data to Dataframe for T
		Xidx_j = data['Xidxj'].values
		mi0 = pd.MultiIndex.from_product([range(t+1,T+1),Xidx])
		mi1 = pd.MultiIndex.from_product([range(t+1,T+1),Xidx_j])
		Data.loc[mi0] = Data.loc[mi1].values		
		
	Data = modify_results(Data,states)
	# mit0 = pd.MultiIndex.from_product([range(t0+1,T+2),Xidx])
	# mit1 = pd.MultiIndex.from_product([range(t0,T+1),Xidx])
	# Data.loc[mit0,'X'] = Data.loc[mit1,'Xj'].values
	# Data.loc[timesteps[0],'X'] = Xval#First State
	# Data.drop(['Xj','Xidxj'],axis=1,inplace=True)
	return Data
	
	
def find_nearest(xj,XX,xsteps):
	"""Find the vector with indices Xidx_j for each possible condition of the system
	characterized by xj with respect to the discretized array Xi_val

	Args:
		xj: array of undiscrete values, which characterize the system and
			indices for which should be found 
		XX: array of arrays, on which Xi_val is based
		xsteps: amount of discretization steps in Xi_val
			
	Returns:
		Xidx_j: vector that contains the index of Xi_val, which value is 
		the nearest to xj 

	"""

	#finding vector of indices i_min for each row of xj and save them to Ind 
	Ind=[]
	lenXX = len(XX)
	Xidx_j = 0
	for i in np.arange(lenXX):
		l=len(XX[i])
		i_min = XX[i].searchsorted(xj[i])
		i_min = np.clip(i_min, 1, l-1)
		left = XX[i][i_min-1]
		right = XX[i][i_min]
		i_min -= xj[i] - left < right - xj[i]
		Ind.append(i_min)
		
		#calculating Xidx_j in dependance of number of states of the system 
		if i == lenXX-1:
			Xidx_j=Xidx_j+Ind[i]
		else:
			Xidx_j = Xidx_j + Ind[i] * xsteps[i+1:].prod() 

	return Xidx_j	
	
def modify_results(Data,states):
	""" Rename columns of result DataFrame Data and drop unnecessary columns
	
	Args:
		Data: result Dataframe with all columns needed for calculation
		states: pandas dataframe where each index represents a state variable
			xmin: minimum value of the state variable
			xmax: maximum value of the state variable
			xstpes: number of discretization steps for this state variable
			
	Returns:
		Data: modified result Dataframe
	
	"""
	
	#for all states, create a column with the name of the state which contains the
	# state value at beginning of each timestep
	for state in states.index:
		col_i = 'Xi_val_'+state #column name which contains state value at beginning of each timestep
		col_j = 'Xj_val_'+state #column name which contains state value at end of each timestep
		

		T = Data.index.levels[0][-2] #last timestep
		T_ = Data.index.levels[0][-1] #additional timestep to store the state value at end of last timestep
		
		Data[state] = Data[col_i] #copy state value at beginning of each timestep
		Data.loc[T_][state] = Data.loc[T,col_j] #add state value at end of last timestep
		Data.drop([col_i,col_j],axis=1,inplace=True) #drop col_i and col_j
	
	Data.drop(['Xidx','Xidxj'],axis=1,inplace=True) #drop 'Xidx' and 'Xidxj'
	
	return Data