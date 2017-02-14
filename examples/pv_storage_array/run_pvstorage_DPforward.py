import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

import pvstorage_model as model 
import prodyn as prd

#####################################
#inputs

#define input file
file = 'pvstorage_data.xlsx'

#load constants, timeseries, decisions U and states from the given input file
#create 
cst,srs,U,states = model.read_data(file)

#import system model function
system=model.pvstorage_model

#timesteps used
timesteps=np.arange(1,49)

#########################################
#Edit initial costs J0
# prodyn allows to define initial costs for the discretized state substeps
# In the forward implementation the DP algorithm starts with the first timestep,
# so the initial costs are given for the first timestep
# At the end of the DP forward algorithm, we can choose the best control schedule
# based on the end state we want. But we can not influence the start state, since it will
# choose the cheapest, unless we edit J0.

#J0 is a numpy array in the same form as the state vector x, where J[i] are initial costs of state x[i]

#In the case of this example, if we don't edit J0, the algorithm will always start with an full storage, 
# since this is cost free energy.

#To force the algorithm to start with an empty storage, we give high negative costs for the empty state
# which is the first index here. Starting from here will then always be the cheapest option for the
# algorithm. (We have to substract this cost at the end in our results to see the real costs!!!)
J0 = np.zeros(states['xsteps'])
J0[0] = -9999.9

#######################################
#Run Dynamic Programming Forward implentation

#run the algorithm with defined states, decisions, constants and series
result = prd.DP_forward(states,U,timesteps,cst,srs,system,J0=J0,verbose=True,t_verbose=10)

#The function returns a pandas DataFrame with 2 index rows "t" and "X_end"
# "t" represents the timestep and "X_end" the end state 
# That means for every possible end state, there is a timeseries containing
# the optimal decision schedule U, the total costs J and the state at the beginning
# of each timestep X plus all variables that are passed in "data" by the system_model
# (here: cost, bat, load_grid)

#Because X is the state at the beginning of each timestep, an additional timestep is 
# added in the results, such that X also contains the state at the end of the last timestep
# For this timestep, all other values (U,J,...) are set to "nan"

#In this case t_start=1 and t_end=48, so index "t" contains also the element 49, where X represents
# the state at the end of timestep 48, and all other columns are "nan"

#######################################
#Access results

# The results for a specific endstate can be accessed as follows
# result.xs(xidx_end,level='Xidx_end'), where xidx_end is the element of Xidx_end which you want to access

#Here we want to get the results where the storage is empty at the end (xidx_end=0)
result0 = result.xs(0,level='Xidx_end')

# To access the different results, notice that the last timestep is not used
bat = result0.loc[timesteps]['bat'].values.astype(np.float)
load_grid = result0.loc[timesteps]['load_grid'].values.astype(np.float)
pv = srs.loc[timesteps]['pv'].values
elec_costs = srs.loc[timesteps]['elec_costs'].values
demand = srs.loc[timesteps]['demand'].values

#The optimal decisions for each timestep can be accesed in the same way
opt_control = result0.loc[timesteps]['U']
print('\n#######################')
print('Optimal control decisions:')
print(opt_control)
print('#######################')

#Only for the states, use all timesteps
# Here the X represents the energy content of the storage
energy = result0['energy']


#######################################
#Prepare results for plotting

grid_import = np.zeros(len(timesteps))
grid_import[load_grid>0] = load_grid[load_grid>0]

grid_export = np.zeros(len(timesteps))
grid_export[load_grid<0] = load_grid[load_grid<0]

bat_charge = np.zeros(len(timesteps))
bat_charge[bat>0] = bat[bat>0]

bat_discharge = np.zeros(len(timesteps))
bat_discharge[bat<0] = bat[bat<0]


####################################
#plot
#####

fig = plt.figure(figsize=(11,7))
gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[3, 1,1])
fs = 18
ms=12

##Subplot 1
ax1 = fig.add_subplot(gs[0])
ax1.plot(timesteps,demand,label='demand',lw=3,color='k')
ax1.stackplot(timesteps,grid_import,-bat_discharge,pv,\
				colors = ['cornflowerblue','lightgreen','yellow'],lw=0)
ax1.stackplot(timesteps,grid_export,-bat_charge,\
				colors = ['navy','green'],lw=0)

ax1.grid()
ax1.tick_params(labelsize=fs-2)
ax1.set_ylabel('Power [kW]',fontsize = fs)
plt.setp(ax1.get_xticklabels(), visible=False)
				
#Legend
ax1.plot([],marker='s',color ='cornflowerblue',label='grid import',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='navy',label='grid export',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='lightgreen',label='from storage',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='green',label='into storage',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='yellow',label='pv',linestyle = 'None',ms = ms)			
ax1.legend(fontsize=fs-2,numpoints = 1,handlelength=1)

##Subplot 2
ax2 = fig.add_subplot(gs[1],sharex=ax1)
ax2.plot(energy,lw=3,color='lightgreen', label = 'Storage energy')

ax2.grid()
ax2.tick_params(labelsize=fs-2)
ax2.set_ylabel('[kWh]',fontsize = fs)
ax2.set_ylim(0,11)
ax2.set_yticks([0,5,10])
plt.setp(ax2.get_xticklabels(), visible=False)

ax2.legend(fontsize=fs-2,numpoints = 1,handlelength=1)

##Subplot 3
ax3 = fig.add_subplot(gs[2],sharex=ax1)
ax3.plot(timesteps,elec_costs,lw=3,color='cornflowerblue', label = 'electricity costs')

ax3.grid()
ax3.tick_params(labelsize=fs-2)
ax3.set_ylabel('[Euro/kWh]',fontsize = fs)
ax3.set_ylim(0,0.6)
ax3.set_yticks([0,0.2,0.4,0.6])
ax3.set_xlabel('time [h]',fontsize = fs)
ax3.set_xlim((0,65))

ax3.legend(fontsize=fs-2,numpoints = 1,handlelength=1, loc='upper left')


plt.show()

