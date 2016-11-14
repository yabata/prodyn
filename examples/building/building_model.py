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
#for 1 state - temperature
###################################	
def building(u,x,t,cst,Srs,Data):
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
    
    l = len(x)
    delay=4
    net = cst['net']
    
    ###################################
    #building input P, previous input P0 and output Y0 arrays for NN's usage
    ###################################
    
    #create 5 inputs for input array P
    hour = Srs.loc[t]['hour']
    solar = Srs.loc[t]['solar']
    T_amb = Srs.loc[t]['T_amb']
    user  = Srs.loc[t]['use_room']
    T_inlet = Srs.loc[t]['T_inlet']
    
    #create 6th input in dependance of current decision
    if u=='heating on':
        massflow = cst['massflow']
    elif u=='heating off':
        massflow = 0
    
    #defining input array P for NN    
    P = np.array([[hour],[solar],[T_amb],[user],[massflow],[T_inlet]],dtype = np.float)
    
    #prepare 5 inputs for P0
    hour0 = Srs.loc[t-delay:t-1]['hour'].values.copy()
    solar0 = Srs.loc[t-delay:t-1]['solar'].values.copy()
    T_amb0 = Srs.loc[t-delay:t-1]['T_amb'].values.copy()
    user0  = Srs.loc[t-delay:t-1]['use_room'].values.copy()
    T_inlet0 = Srs.loc[t-delay:t-1]['T_inlet'].values.copy()
    
    #defining initial values
    x_j = np.zeros(l)
    P_th = np.zeros(l)
    costx = np.zeros(l)
    
    #initializing the starting position of optimization     
    t0 = cst['t0']
    
    #loop for every possible state of x
    for i,xi in enumerate(x):
        #prepare 6th input for P0 and 2 outputs for Y0
        if t-delay<t0:
            #take all values for P0 and Y0 from timeseries            
            if Data is None or t==t0:
                T_room0 = Srs.loc[t-delay:t-1]['T_room'].values.copy()
                P_th0 = Srs.loc[t-delay:t-1]['P_th'].values.copy()
                massflow0 = Srs.loc[t-delay:t-1]['massflow'].values.copy()
            
            #take part of values from timeseries and part from big Data            
            else:
                tx = t-t0
                T_room0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['T_room'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['T_room'].values.copy()])
                P_th0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['P_th'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['P_th'].values.copy()])
                massflow0 = np.concatenate([Srs.loc[t-delay:t-tx-1]['massflow'].values.copy(),Data.loc[t-tx-1:t-1].xs(i,level='Xidx_end')['massflow'].values.copy()])
        
        #take all values for P0 and Y0 from big Data
        else:
            T_room0 =Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['T_room'].values.copy()
            P_th0 = Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['P_th'].values.copy()
            massflow0 = Data.loc[t-delay:t-1].xs(i,level='Xidx_end')['massflow'].values.copy() 

         
        #correcting the last value for T_room0
        T_room0[-1] = xi
        
        #Create P0 and Y0
        P0 = np.array([hour0,solar0,T_amb0,user0,massflow0,T_inlet0],dtype = np.float)
        Y0 = np.array([T_room0,P_th0],dtype = np.float)
        
        #run NN for one timestep        
        if np.any(P0!=P0) or np.any(Y0!=Y0):
            #if P0 or Y0 not valid use valid values and apply penalty costs
            costx[i] = 1000*10
            x_j[i] = xi
            P_th[i] = 0
        
        else:
            x_j[i],P_th[i] = prn.NNOut(P,net,P0=P0,Y0=Y0)
        
        if x_j[i] != x_j[i] or P_th[i] != P_th[i]:
            pdb.set_trace()
        
        #print(T_room0,u, x_j[i])
    

    ###################################
    #calculating all costs
    ###################################    
    
    #selecting borders for allowed Troom
    Tmax = Srs.loc[t]['Tmax']
    Tmin = Srs.loc[t]['Tmin']    
    
    #calculate penalty costs
    costx = (x_j>Tmax)*1000 + (x_j<Tmin)*1000+costx
    x_j = (x_j<x[0])*x[0]\
            +(x_j>x[-1])*x[-1]\
            +((x_j>=x[0])&(x_j<=x[-1]))*x_j
    
    #calculate costs
    P_el = P_th*T_inlet/(T_inlet-T_amb)
    cost = P_el * Srs.loc[t]['price_elec']*0.25 + costx
    
    #define results to be put in data
    data = pd.DataFrame(index = np.arange(l))
    data['P_th'] = P_th
    data['P_el'] = P_el
    data['T_room'] = x_j
    data['massflow'] = massflow
    data['cost'] = cost
    data['costx'] = costx
        
    
    return cost, x_j, data
    
