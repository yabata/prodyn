import pandas as pd
import numpy as np
import prodyn as prd
import matplotlib.pyplot as plt

#States
#####
#At first, we describe the state variable "battery" of our system by defining
#its minimum and maximum value and the number of dicretization steps
#therefore we create a pandas DataFrame where the index represents the states
#(here only "battery") and the columns ['xmin','xmax','xsteps']
states = pd.DataFrame(index=['battery'], columns=['xmin','xmax','xsteps'])

#we define a battery with 1 kWh capacity and define our steps to be 2 
#(0kWh, 1kWh)
states.loc['battery','xmin'] = 0
states.loc['battery','xmax'] = 1
states.loc['battery','xsteps'] = 2

#Controls (decisions)
#####
#Now we define the possible control signals (decisions) of our system. Therefore 
#we simply create a list containing all possible control signals. This can be 
#numbers, or strings or any other type of variable, as long as the system model 
#is defined such that it can handle the defined control signals. 
#In our case we choose integers to represent our control signal. 1 for charging
#the battery, 0 for doing nothing, -1 for dischraging it.
controls = [-1,0,1] #The set of control signals/decisions (often called "U")

#timesteps
#####
#We define the timesteps for our optimization. In this case we want to optimize
#for 3 hours, so we define a numpy array [0,1,2,3]
timesteps=np.arange(1,4)


#Constant Parameters
######
#We can define two variables that contain constant parameters we can access
#within our system function. Both variables can be defined as any type 
#(dict, pandas DataFrame, list, array, constant,....)
#For a better readability I usually define one variable containing all the 
#time independent constants and one variable containing a timeseries
#For this simple example, we define a constant variable as a dict, containing 
#only the maximum capacity of the battery and a pandas DataFrame containing 
#the electricity price curve (with timestpes as index)
constants = {'max_cap': 1}
timeseries = pd.DataFrame(index=timesteps, columns=['el. price'])
#As an incentive to use the storage we define a price curve with low costs 
#in the beginning and high costs at the end of the day
timeseries['el. price'] = np.array(
                            [1,2,3])
                            
#System model
#####
#Now we define the function describing our system. The function calculates for 
#every possible state x and a given control signal u the following state x_j
#and the costs for going from x to x_j.
#Allowing prodyn to use a function, it has to be defined following some rules
#for the inputs and outputs.
#The function has to be defined with 6 inputs (u,x,t,cst,Srs,Data), where:
# u     is the current control signal (decision) within possible decisions U
#       in our case u is one element of our defined controls, either -1, 1 or 0
# x     is a numpy array with dicretized steps of the state
#       in our case x = [0, 1, 2, 3, 4, 5]
# t     is the current timestep
# cst   is one of our constant variables we defined, in our case constants
# Srs   is the other constant variables we defined, in our case timeseries
# Data  is a Dataframe created by prodyn containing previous results of the
#       optimization. It allows us to access results from already solved
#       timesteps. For this example, Data will not be used. How to use it
#       is explained in another example.
#
#The function has to return 3 outputs (cost, x_j, data), where:
# cost  is a numpy array of the same size as x containing the cost for going
#       from x to x_j when using control signal u
# x_j   is the following state of x (numpy array of the same size as x)
# data  contains additional variables that will be saved in the result DataFrame
#       "Data", which also can be used in the following timesteps. 
#       data has to be defined as a pandas DataFrame with index x and 
#       a column for each variable that should be saved. It is not necessary
#       to use data, but it has to be defined as a (empty) pandas DataFrame 
#       with index=x

#Of course, the names of the inputs and outputs can be changed, but the order 
#and typ of the variables have be to as described

def storage_model(u,x,t,cst,Srs,Data):

    #for our simple model we initialize the arrays cost and x_j with zeros
    #the arrays have to be the same size as x
    l = len(x) #number discretization steps of x    
    cost = np.zeros(l)
    x_j = np.zeros(l)
    
    #we inizialize an additional array penelty_cost, which we use in our model
    penalty_cost = np.zeros(l)
    
    #We have to calculate the following state x_j and the cost using the control
    #signal u for every possible state beginning state x_i (every element of x)
    #In this example we use a loop to do this: 
    
    for i,x_i in enumerate(x):
        
        #prodyn always minimizes the total costs over all timesteps
        #here we define the costs to be positive when we charge the battery
        #because we have to buy electricity and negative when we discharge it,
        #because then we sell electricity. The costs are zero, if we do nothing.
        #In this simple example we can charge or discharge with 1 kW per 
        # timestep (1 hour), so we increase or decrease the storage content by 
        # 1kWh. So the costs are defined by the electricity price in the current
        #timestep. By muliplying the price with the control signal u, which is
        #-1 for discharging, 0 for doing nothing and 1 for discharging, the cost
        #becomes negative, zero or positive, as we defined it.
        cost[i] = (u * Srs.loc[t]['el. price'])
        #as just described, the battery energy content, which is our state 
        #variable increase by 1 kWh when charging the battery (u=1), decreases 
        #by 1 kWh when discharging it (u=-1) and stays the same when doing 
        #nothing (u=0)
        x_j[i] = x_i + u
        
        #With our definition above, it would be psooible that an already fully
        #charged battery with 5kWh energy content could be charged up to 6kWh
        #If this happens, change the energy content back to 5 kWh ('max_cap')
        #and we set the penalty costs to a high value (999)
        if x_j[i] > cst['max_cap']:
            x_j[i] = cst['max_cap']
            penalty_cost[i] = 999
        #we do the same, if the energy content goes below 0 kWh.
        elif x_j[i] < 0:
            x_j[i] = 0
            penalty_cost[i] = 999
        
        #Now we add the penalty costs to the cost. High costs prevent that this
        #decision (control signal) will be chosen by the algorithm to be part of
        #the solution with minimal costs at the end
        cost[i] = cost[i] + penalty_cost[i]
    
    #We have to create a pandas DataFrame with index=x for the output (even
    #if we don't use it)
    data = pd.DataFrame(index = x)
    #We create the row 'cost', where we save the costs for every x when applying
    #control signal u at the current timestep t. Tzhe costs will be saved in the
    # results and we cann acess them after the problem is solved.
    data['cost'] = cost
    
    return cost,x_j,data

#Solving the problem and accessing the solution using prodyn
#####

#Prodyn can be used to find the control sequence that leads to the minimum
#total costs over the optimization horizon (timesteps) using the dynamic
#programming algorithm.
#prodyn has two different implementations of the DP algorithm.
#DP_forward solves the problem "forward in time", which means the algorithm
#starts at the first timestep, while the DP_backward starts at the last timestep
#and solves the problem "backward in time". The forward algorithm has the 
#advantage, that states and calculated parameters from previous timesteps
#can be used in the model, which might be necessary to model the problem.
#The backward algorithm only can access "future" states and parameters, but
#because it is usually faster and therefore should be chosen if past information
#is not necessary to model the problem. Both algorithms are described more
#detailed in the chapter "how prodyn works".

#DP_forward
###


result_fw = prd.DP_forward(states,controls,timesteps,constants,timeseries,
                           storage_model,verbose=True,t_verbose=1)

#Here we want to get the results where the storage is empty at the end (Xidx_end=0)
result0_fw = result_fw.xs(0,level='Xidx_end')

#DP_backward
###
result_bw = prd.DP_backward(states,controls,timesteps,constants,timeseries,
                            storage_model,verbose=True,t_verbose=1)

#Here we want to get the results where the storage is empty at the beginning 
#(Xidx_start=0)
result0_bw = result_bw.xs(0,level='Xidx_start')
#And compare it to the results where the storage is full at the beginning 
#(Xidx_start=1)
result5_bw = result_bw.xs(1,level='Xidx_start')

plt.plot(result0_fw['battery'],'b')
plt.plot(result5_bw['battery'],'g-.')
plt.plot(result0_bw['battery'],'r:')
plt.show()
