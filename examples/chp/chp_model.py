import pandas as pd
import numpy as np
import pdb
	
def read_data(file):
    """Read data about the system from the excel file and assign 
    it to different parameters
    
    Args:
            file: excel file, which stores all data about considered system
            
    Returns:
            cst: constants, which describe the system
            srs: parameters, which are variable with time
            U: list of possible decisions
            states: values, which set number of states and characterize 
            all possible ones
    """

    xls = pd.ExcelFile(file)
    states = xls.parse('DP-States',index_col=[0])
    cst = xls.parse('Constants',index_col=[0])['Value']
    srs = xls.parse('Time-Series',index_col=[0])
    U = xls.parse('DP-Decisions',index_col=[0])['Decisions'].values
    return cst,srs,U,states
    

 
###################################
#for 2 states
###################################	
def chp(u,x,t,cst,srs,Data):
	"""For current timestep t and for current decision u transition from 
	actual timestep i to the following timestep j is simulated for all
	possible states, which are stored in x.

	Costs of this transition and array of states after it are calculated.

	Args:
			u: decision from list of possible ones
			x: array, where all possible system states are stored
			t: actual timestep i
			cst: constants needed for calculation
			srs: values of needed timeseries

	Returns:
			cost: costs at timestep i
			x_j: array with states at timestep j after transition due to
					decision u
			data: dataframe, which keeps additional infromation about 
					transition from i to j
	"""

	l = len(x[0])

	#defining max and min content values for both storages  
	bat_min=x[0][0]
	bat_max=x[0][len(x[0])-1]

	heatsto_min=x[1][0]
	heatsto_max=x[1][len(x[1])-1]

	costx=0
	#############################
	#Calculations of penalty costs and amount of charge/discharge of the 
	#storages for all u decisions    
	#############################  
	
	#CHP is off
	###########
	#El and heat demand is covered by storages if possible and else by the grid/gas boiler.
	if u == 'off':
		

		gas_chp=0
	  
		bat_pow = -np.ones(l)*np.min([cst['battery_Pmax'],srs.loc[t]['el_demand']])
		bat_en = x[0] + bat_pow
		bat_pow[bat_en<bat_min] = bat_min - x[0][bat_en<bat_min]
		bat_en = x[0] + bat_pow
		el_demand_ext = srs.loc[t]['el_demand'] + bat_pow

		heatsto_pow = -np.ones(l)*np.min([cst['heat-storage_Pmax'],srs.loc[t]['heat_demand']])
		heatsto_en = x[1] + heatsto_pow
		heatsto_pow[heatsto_en<heatsto_min] = heatsto_min - x[1][heatsto_en<heatsto_min]
		heatsto_en = x[1] + heatsto_pow
		heat_demand_boiler = srs.loc[t]['heat_demand'] + heatsto_pow
		
		
	#CHP is on
	##########
	#Surplus of electricity is stored to the battery and sold to the grid if it is full. 
	#Surplus of heat is stored. If storage is full penalty costs are applied.
	elif u == 'on':
		
		gas_chp = cst['chp_Pmax_el']/cst['chp_eff_el']
		
		el_demand_bat = srs.loc[t]['el_demand'] - cst['chp_Pmax_el']
		
		if el_demand_bat > 0:
			# el_demand_bat is covered by the battery if possible
			bat_pow = -np.ones(l)*np.min([cst['battery_Pmax'],el_demand_bat])
			bat_en = x[0] + bat_pow
			bat_pow[bat_en<bat_min] = bat_min - x[0][bat_en<bat_min]
			bat_en = x[0] + bat_pow
			el_demand_ext = srs.loc[t]['el_demand'] + bat_pow - cst['chp_Pmax_el']
		else:
			# battery is charged with surplus
			bat_pow = np.ones(l)*np.min([cst['battery_Pmax'],-el_demand_bat])
			bat_en = x[0] + bat_pow
			bat_pow[bat_en>bat_max] = bat_max - x[0][bat_en>bat_max]
			bat_en = x[0] + bat_pow
			el_demand_ext = srs.loc[t]['el_demand'] + bat_pow - cst['chp_Pmax_el']
		
		chp_heat = (cst['chp_Pmax_el']/cst['chp_eff_el'])*cst['chp_eff_th']
		heat_demand_sto = srs.loc[t]['heat_demand'] - chp_heat
		
		if heat_demand_sto > 0:
			# heat_demand_sto is covered by the heat storage if possible
			heatsto_pow = -np.ones(l)*np.min([cst['heat-storage_Pmax'],heat_demand_sto])
			heatsto_en = x[1] + heatsto_pow
			heatsto_pow[heatsto_en<heatsto_min] = heatsto_min - x[1][heatsto_en<heatsto_min]
			heatsto_en = x[1] + heatsto_pow
			heat_demand_boiler = srs.loc[t]['heat_demand'] + heatsto_pow - chp_heat

		else:
			# heat storage is charged with surplus
			heatsto_pow = np.ones(l)*np.min([cst['heat-storage_Pmax'],-heat_demand_sto])
			heatsto_en = x[1] + heatsto_pow
			heatsto_pow[heatsto_en>heatsto_max] = heatsto_max - x[1][heatsto_en>heatsto_max]
			heatsto_en = x[1] + heatsto_pow
			heat_demand_boiler = srs.loc[t]['heat_demand'] + heatsto_pow - chp_heat
			costx = costx + (heat_demand_boiler<0)*9999


	x_j = np.array([bat_en,heatsto_en])
	
	
	gas_boiler = heat_demand_boiler/cst['boiler_eff']
	gas_ext = gas_boiler + gas_chp
	
	#calculating costs of transition
	cost = el_demand_ext * srs.loc[t]['el_cost'] *(el_demand_ext>0)\
			+ el_demand_ext * srs.loc[t]['el_feed-in'] *(el_demand_ext<0) \
			+gas_ext * cst['gas_cost'] + costx

	#saving all data
	data = pd.DataFrame(index = np.arange(l))
	data['el_demand_ext'] = el_demand_ext
	data['bat_pow'] = bat_pow
	data['heatsto_pow'] = heatsto_pow
	data['heat_demand_boiler'] = heat_demand_boiler

	# pdb.set_trace()
	return cost,x_j,data
    
 

    
    
	



