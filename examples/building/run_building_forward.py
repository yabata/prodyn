import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn

import building_model as model
import prodyn as prd


#define input excel file
file = 'building_data.xlsx'

#define constants, timeseries, list of possible decisions U and
#timesteps, which are needed for the model
cst,srs,U,states = model.read_data(file)

#add some data to timeseries
srs['massflow'] = 0
srs['P_th'] = 0
srs['T_room'] = 20

#define timesteps, on which optimization will be realized
timesteps=np.arange(cst['t_start'],cst['t_end'])

#define NN 
net = prn.loadNN('testNN_aprbs.csv') #use pre-trained NN
cst['net'] = net

#choose timestep, from which optimization will start
cst['t0']=4

#creating array of initial terminal costs J0 
xsteps=np.prod(states['xsteps'].values)
J0 = np.zeros(xsteps)
J0[0] = -9999.9

#define function for simulation that calculates costs and next state
system=model.building

#optimize with DP forward algorithm
result = prd.DP_forward(states,U,timesteps,cst,srs,system,J0=J0,verbose=True,t_verbose=5)


#find cost minimal solution from result
i_mincost = result.loc[cst['t_end']-1]['J'].idxmin()
opt_result = result.xs(i_mincost,level='Xidx_end')

######################################################
#extracting interesting parameters and plotting them
######################################################

#extract parameters from cost minimal solution
best_massflow=opt_result['massflow'].values[:-1]
Troom=opt_result['T_room'].values[:-1]
Pel=opt_result['P_el'].values[:-1]

#summing first initial timesteps and timesteps, which are involved in optimisation
Troom=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['T_room'],Troom))
Pel=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['P_th'],Pel))

#selecting borders for allowed Troom
Tmax = srs.loc[timesteps[0]-4:timesteps[-1]]['Tmax']
Tmin = srs.loc[timesteps[0]-4:timesteps[-1]]['Tmin']

########################################################
#plotting
########################################################
fig = plt.figure(figsize=[11,9])
ls=14
bs=16    
xlim=timesteps[-1]
    
ax0 = fig.add_subplot(211)
ax1 = fig.add_subplot(212,sharex=ax0)

#first plot
ax0.set_title('Results of 1-day (4-100) optimization')
ax0.plot(Troom,lw = 2,label='$T_{room}$')
ax0.set_ylabel('$T, [$^\circ$C]$',fontsize = bs)
ax0.set_xlim([0,xlim])
ax0.set_xlabel('$timesteps$',fontsize = bs)
ax0.plot(Tmin,color='r',lw=1,label='$T_{min}$')
ax0.plot(Tmax,color='r',lw=1,label='$T_{max}$')
ax0.tick_params(axis='x',labelsize=ls)
ax0.tick_params(axis='y',labelsize=ls)
ax0.legend(fontsize=bs,loc=1)
ax0.grid()
 
#second plot   
ax1.plot(Pel,lw = 2)
ax1.set_ylabel('$P_{el}, [kW]$',fontsize = bs)
ax1.set_xlim([0,xlim])
ax1.set_xlabel('$timesteps$',fontsize = bs)
ax1.tick_params(axis='x',labelsize=ls)
ax1.tick_params(axis='y',labelsize=ls)
ax1.grid()
    
fig.tight_layout()
    
plt.show()

