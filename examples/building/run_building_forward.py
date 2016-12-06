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
net = prn.loadNN('NN_building.csv') #use pre-trained NN
cst['net'] = net

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

#taking values for price electricity and solar radition
price_elec=srs.loc[timesteps[0]-4:timesteps[-1]]['price_elec']
solar=srs.loc[timesteps[0]-4:timesteps[-1]]['solar']

########################################################
#plotting
########################################################
fig = plt.figure(figsize=[11,9])
ls=14
bs=16    

#prepairing array time, which transforms initial and optimisation timesteps in hours
timesteps0=np.arange(timesteps[0]-4,timesteps[0])
time=np.concatenate((timesteps0,timesteps)).astype('float')/4
    
ax0 = fig.add_subplot(211)
ax1=ax0.twinx()

ax2 = fig.add_subplot(212,sharex=ax0)
ax3=ax2.twinx()

#first plot
ax0.set_title('Results of 1-day (4-100) optimization')
lns1=ax0.plot(time, Troom, lw = 2,label='$T_{room}$')
ax0.set_ylabel(r'$T, [^\circ C]$',fontsize = bs)
ax0.set_ylim([15,27])
ax0.set_xlabel('$time, [h]$',fontsize = bs)
lns2=ax0.plot(time,Tmax,color='r',lw=1,label='$T_{max}$')
lns3=ax0.plot(time, Tmin,color='r',lw=1,label='$T_{min}$')
ax0.tick_params(axis='x',labelsize=ls)
ax0.tick_params(axis='y',labelsize=ls)
#ax0.legend(fontsize=bs,loc=9, ncol=3)
ax0.grid()

lns4=ax1.plot(time,solar,color = 'g', lw = 2, label='$solar$')
ax1.set_ylabel('$solar$ $rad, [kJ/hm^2]$',fontsize = bs)
ax1.set_ylim([0,2000])
ax1.tick_params(axis='y',labelsize=ls)
lnsa = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lnsa]
ax0.legend(lnsa, labs,fontsize=bs, loc=9,ncol=4)

 
#second plot   
lns1=ax2.plot(time, Pel,lw = 2,label='$P_{el}$')
ax2.set_ylabel('$P_{el}, [kW]$',fontsize = bs)
ax2.set_xlabel('$time, [h]$',fontsize = bs)
ax2.set_ylim([-0.5,1.5])
ax2.tick_params(axis='x',labelsize=ls)
ax2.tick_params(axis='y',labelsize=ls)
ax2.grid()

lns2=ax3.plot(time,price_elec*100,color = 'y', lw = 2, label='$price$ $elec$')
ax3.set_ylabel('$elec$ $price, [cent/kWh]$',fontsize = bs)
ax3.set_ylim([15,80])
ax3.tick_params(axis='y',labelsize=ls)
lnsb = lns1+lns2
labs = [l.get_label() for l in lnsb]
ax2.legend(lnsb, labs,fontsize=bs, loc=9,ncol=2)
    
#fig.tight_layout()
    
plt.show()

