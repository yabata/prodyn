import pandas as pd
import numpy as np
import pyrenn as prn
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
#for 2 states - temperature and heat-storage
###################################	
def building_with_storage(u,x,t,cst,Srs,Data):
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

	

    ###############################################    
    #Defining T_room and P_th for timestep j
    #with using pre-trained NN    
    ###############################################

    l = len(x[0])
    
    delay=4	
    net = cst['net']
    
    #create 5 inputs for input array P
    hour = Srs.loc[t]['hour']
    solar = Srs.loc[t]['solar']
    T_amb = Srs.loc[t]['T_amb']
    user  = Srs.loc[t]['use_room']
    T_inlet = Srs.loc[t]['T_inlet']

    #create 6th input in dependance of current decision
    if u=='pump on/storage on' or u=='pump off/storage on':
        massflow = cst['massflow']
    elif u=='pump off/storage off' or u=='pump on/storage off':
        massflow = 0

    #defining input array P for NN    
    P = np.array([[hour],[solar],[T_amb],[user],[massflow],[T_inlet]],dtype = np.float)

    #prepare 5 inputs for P0
    hour0 = Srs.loc[t-delay:t-1]['hour'].values.copy()
    solar0 = Srs.loc[t-delay:t-1]['solar'].values.copy()
    T_amb0 = Srs.loc[t-delay:t-1]['T_amb'].values.copy()
    user0  = Srs.loc[t-delay:t-1]['use_room'].values.copy()
    T_inlet0 = Srs.loc[t-delay:t-1]['T_inlet'].values.copy()

    #defining initial values, which are used inside the loop
    T_roomj = np.zeros(l)
    P_th = np.zeros(l)
    
    #defining initial values, which are used outside the loop
    E_j = np.zeros(l)
    P_el = np.zeros(l)
    costx = np.zeros(l)
    
    #loop for every possible temperature state 
    for i,x1 in enumerate(x[0]):
        #prepare 6th input for P0 and 2 outputs for Y0
        if t-delay<cst['t_start']:
            #take all values for P0 and Y0 from timeseries            
		if Data is None or t==cst['t_start']:
			T_room0 = np.ones(delay) * x1
			P_th0 = Srs.loc[t-delay:t-1]['P_th'].values.copy()
			massflow0 = Srs.loc[t-delay:t-1]['massflow'].values.copy()
			
		#take part of values from timeseries and part from big Data            
		else:
			tx = t-cst['t_start']
			T_room0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['T_room'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['T_room'].values.copy()])
			P_th0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['P_th'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['P_th'].values.copy()])
			massflow0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['massflow'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['massflow'].values.copy()])
       
        #take all values for P0 and Y0 from big Data
        else:
		T_room0 =Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['T_room'].values.copy()
		P_th0 = Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['P_th'].values.copy()
		massflow0 = Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['massflow'].values.copy() 

		
        #Create P0 and Y0
        P0 = np.array([hour0,solar0,T_amb0,user0,massflow0,T_inlet0],dtype = np.float)
        Y0 = np.array([T_room0,P_th0],dtype = np.float)
        
               
        #run NN for one timestep        
        if np.any(P0!=P0) or np.any(Y0!=Y0) or np.any(Y0>1000):
		#if P0 or Y0 not valid use valid values and apply penalty costs
		costx[i] = 1000*10
		T_roomj[i] = x1
		P_th[i] = 0
        
        else:
		T_roomj[i],P_th[i] = prn.NNOut(P,net,P0=P0,Y0=Y0)
		
        if T_roomj[i] != T_roomj[i] or P_th[i] != P_th[i]:
		pdb.set_trace()
 
        
    #calculating heat-storage state in dependance of chosen decision 
    P_hp = 2
    if u=='pump on/storage on':
		E_j=x[1]
		P_el = 3*P_th			
    
    elif u=='pump on/storage off':
		E_j=x[1]+P_hp*3*0.25
		P_el = P_hp
			
    elif u=='pump off/storage on':
		E_j=x[1]-P_th*0.25
		P_el = 0

    elif u=='pump off/storage off':
		E_j=x[1]
		P_el = 0		
		 
    costx = costx + 99999*(E_j<x[1][0]) + 99999*(E_j>x[1][-1])
		 
	

    ###############################################
    #Building array x_j for the timestep j and
    #calculating all costs for transition from i to j
    ###############################################    

    #building x_j
    x_j=np.vstack((T_roomj,E_j))

    #selecting borders for allowed Troom
    Tmax = Srs.loc[t]['Tmax']
    Tmin = Srs.loc[t]['Tmin']

    #selecting borders for possible energy content of heat storage E
    Emax=x[1][-1]
    Emin=x[1][0]

    #Calculate penalty costs
    costx = (x_j[0]>Tmax)*(x_j[0]-Tmax)**2*99999 + (x_j[0]<Tmin)*(x_j[0]<Tmin)**2*9999\
            +(x_j[1]>Emax)*99999 + (x_j[1]<Emin)*99999\
            +costx

    #correcting x_j    
    x_j[0] = np.clip(x_j[0],x[0][0],x[0][-1])
    x_j[1] = np.clip(x_j[1],x[1][0],x[1][-1])           

    #Calculate costs
    cost = P_el * Srs.loc[t]['price_elec']*0.25 + costx

    #Define results to be put in Data
    data = pd.DataFrame(index = np.arange(l))
    data['P_th'] = P_th
    data['P_el'] = P_el
    data['T_room'] = x_j[0]
    data['E'] = x_j[1]
    data['massflow'] = massflow
    data['cost'] = cost
    data['costx'] = costx
	
    return cost, x_j, data
    
