import pandas as pd
import numpy as np
import time
import pdb
	
def read_data(file):

	xls = pd.ExcelFile(file)
	cst = xls.parse('Constants',index_col=[0])['Value']
	srs = xls.parse('Time-Series',index_col=[0])
	U = xls.parse('DP-Decisions',index_col=[0])['Decisions'].values
	states = xls.parse('DP-States').loc[0]
	return cst,srs,U,states

	
def system_model(u,x,t,cst,Srs,Data):
	"""Simulate actual timestep for decision u and all states x

	Calculate the costs c for actual timestep i
	Calculate the state for following timestep j
	
	Args:
		u: current decision u within possible decisions U
		x: numpy array with dicretized stpes of the state
		t: current timestep
		cst: dict or pandas Series containing constants needed for system calculations
		srs: pandas Dataframe containing timeseries needed for system caluclation
		Data: Dataframe containing previous results
		
	Returns:
		c: costs at timestep i
		x_j: state i = at the beginning of timestep j
	"""
	#state x represents energy content of the battery (in kWh)
	
	srs=Srs.loc[t] #time-series data of current timestep
	l = len(x) #number discretization steps of x
	xmin = x[0] #minimum Value of x
	xmax = x[-1] #maximum Valus of x
	
	####
	#Initialize results
	cost = np.zeros(l)
	x_j = np.zeros(l)
	bat = np.zeros(l)
	load_grid = np.zeros(l)
	
	for i,xi in enumerate(x):
	
		#residual load: load that has to be covered(+) or surplus power (-)
		res = Srs.loc[t]['demand'] - Srs.loc[t]['pv']
		
		penalty_costs = 0 #initialize penalty costs
		
		#########################
		#Decisions
		
		#Normal
		if u == 'normal':
		
			bat[i] = 0 #battery power is zero
			penalty_costs = 0 # no penalty costs
			x_j[i] = xi #energy content of battery does not change
		
		#Charge
		elif u == 'charge':
			
			if res>0:
				#charging is not possible for negative residual load -> penalty costs
				penalty_costs = 999
			elif xi == xmax:
				penalty_costs = 999# charging is not possible fo a full storage (x==xmax) -> penalty costs
			else:

				bat[i] = np.min([cst['P_max'],-res]) # charging power is negative residual load (unless it is not greater than maximum battery power)
				x_j[i] = xi + bat[i] #next state (energy content) after charging
				
				if x_j[i]>xmax:
					#If new energy content x_j is greater than maximum energy content, update bat
					#such that x_j = xmax
					bat[i] = xmax - xi
					x_j[i] = xmax # x_j then is xmax
			
			
		#Discharge
		elif u == 'discharge':
			if res<0:
				#discharging is not possible for positive residual load -> penalty costs
				penalty_costs = 999
			elif xi == xmin:
				penalty_costs = 999 #discharging is not possible fo a empty storage (x==xmin) -> penalty costs
			else:
			
				bat[i] = np.max([-cst['P_max'],-res]) # discharging power is negative residual load (unless it is not greater than maximum battery power)

				x_j[i] = xi + bat[i] #next state (energy content) after charging
			
				if x_j[i]<xmin:
					#If new energy content x_j is lower than minimum energy content, update bat
					#such that x_j = xmin
					bat[i] = xmin - xi
					x_j[i] = xmin # x_j then is xmin
		
		
		#Calculate load that has to be covered by (positive) or fed into the grid (negative)
		load_grid[i] = res + bat[i]
		
		if load_grid[i]<0:
			#if power is fed into the grid, a feed-in tariff of 0.11 Euro/kWh is assumed
			cost[i] = load_grid[i] * 0.11
		else:
			#cost for power import from the grid are given in the time-series 
			cost[i] = load_grid[i] * Srs.loc[t]['elec_costs']
		
		#Add penalty costs
		cost[i] = cost[i] + penalty_costs
	
	#Create DataFrame to save information
	#these will be added to Data and can be used within the next timestep
	data = pd.DataFrame(index = x)
	data['load_grid'] = load_grid
	data['bat'] = bat
	data['cost'] = cost
	
	#The system function has to return these three outputs
	# cost is a numpy array with total costs for every state
	# x_j is the following state of x, for the current decision
	# data contains additional variables that will be saved in "Data" (input) and can be used in the following timesteps
	return cost,x_j,data
	

