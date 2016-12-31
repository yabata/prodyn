import numpy as np
import matplotlib.pyplot as plt
import pyrenn as prn

import building_with_storage_model as model
import prodyn as prd

#define input excel file
file = 'building_with_storage_data.xlsx'

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
idx = prd.find_index(np.array([20,0]),states)
J0[idx] = -9999.9

#define function for simulation that calculates costs and next state
system=model.building_with_storage

#optimize with DP forward algorithm
result = prd.DP_forward(states,U,timesteps,cst,srs,system,J0=J0,verbose=True)


#find cost minimal solution from result
i_mincost = result.loc[cst['t_end']-1]['J'].idxmin()
opt_result = result.xs(i_mincost,level='Xidx_end')

#find solution with empty heat-storage from result
#opt_result = result.xs(0,level='Xidx_end')

######################################################
#extracting interesting parameters and plotting them
######################################################
best_massflow=opt_result['massflow'].values[:-1]
Troom=opt_result['T_room'].values[:-1]
Pel=opt_result['P_el'].values[:-1]
Pth=opt_result['P_th'].values[:-1]
E=opt_result['heat-storage'].values[:-1]

#summing first initial timesteps and timesteps, which are involved in optimisation
Troom=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['T_room'],Troom))
Pel=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['P_th'],Pel))
Pth=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['P_th'],Pth))
E=np.concatenate((np.zeros(4),E))

#selecting borders for allowed Troom
Tmax = srs.loc[timesteps[0]-4:timesteps[-1]]['Tmax']
Tmin = srs.loc[timesteps[0]-4:timesteps[-1]]['Tmin']

#taking values for price electricity and solar radition
price_elec=srs.loc[timesteps[0]-4:timesteps[-1]]['price_elec']
solar=srs.loc[timesteps[0]-4:timesteps[-1]]['solar']


########################################################
#plotting
########################################################
fig1 = plt.figure(figsize=[11,11])
ls=14
bs=16    

#prepairing array time, which transforms initial and optimisation timesteps in hours
timesteps0=np.arange(timesteps[0]-4,timesteps[0])
time=np.concatenate((timesteps0,timesteps)).astype('float')/4
    
ax0 = fig1.add_subplot(311)
ax1=ax0.twinx()

ax2 = fig1.add_subplot(312,sharex=ax0)
ax3=ax2.twinx()

ax4 = fig1.add_subplot(313,sharex=ax0)

##################first plot
#1st y-axis
ax0.set_title('Results of 1-day (19-44 hours) optimization')
lns1=ax0.plot(time, Troom, lw = 2,label='$T_{room}$')
ax0.set_ylabel(r'$T, [^\circ C]$',fontsize = ls)
ax0.set_ylim([15,30])
ax0.set_xlabel('$time, [h]$',fontsize = ls)
lns2=ax0.plot(time,Tmax,color='r',lw=1,label='$T_{max}$')
lns3=ax0.plot(time, Tmin,color='r',lw=1,label='$T_{min}$')
ax0.tick_params(axis='x',labelsize=ls-2)
ax0.tick_params(axis='y',labelsize=ls-2)
#ax0.legend(fontsize=bs,loc=9, ncol=3)
ax0.grid()

#2nd y-axis 
lns4=ax1.plot(time,solar/1000,color = 'g', lw = 2, label='$solar$')
ax1.set_ylabel('$solar$ $rad, [MJ/hm^2]$',fontsize = ls)
ax1.set_ylim([0,2])
ax1.tick_params(axis='y',labelsize=ls-2)

#legend
lnsa = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lnsa]
ax0.legend(lnsa, labs,fontsize=ls, loc=9,ncol=4)

 
#####################second plot   
#1st y-axis
lns5=ax2.plot(time, Pel, color='blueviolet', lw = 2,label='$P_{el}$')
ax2.set_ylabel('$P_{el}, [kW]$',fontsize = ls)
ax2.set_xlabel('$time, [h]$',fontsize = ls)
ax2.set_ylim([-1,7])
ax2.tick_params(axis='x',labelsize=ls-2)
ax2.tick_params(axis='y',labelsize=ls-2)
ax2.grid()

#2nd y-axis
lns6=ax3.plot(time, Pth,color='orangered',lw = 2,label='$P_{th}$')
ax3.set_ylabel('$P_{th}, [kW]$',fontsize = ls)
ax3.set_ylim([0,2])
ax3.tick_params(axis='y',labelsize=ls-2)

#legend
lnsb = lns5+lns6
labs = [l.get_label() for l in lnsb]
ax2.legend(lnsb, labs,fontsize=ls, loc=1,ncol=2)


# lns2=ax3.plot(time,price_elec*100,color = 'k', lw = 2, label='$price$ $elec$')
# ax3.set_ylabel('$elec$ $price, [cent/kWh]$',fontsize = bs)
# ax3.set_ylim([15,80])
# ax3.tick_params(axis='y',labelsize=ls)
# lnsb = lns1+lns2
# labs = [l.get_label() for l in lnsb]
# ax2.legend(lnsb, labs,fontsize=bs, loc=9,ncol=2)

#third plot
ax4.plot(time, E,color='c', lw = 2)
ax4.set_ylabel('$E, [kWh]$',fontsize = ls)
# ax4.set_ylim([0,1])
ax4.set_xlabel('$time, [h]$',fontsize = ls)
ax4.tick_params(axis='x',labelsize=ls-2)
ax4.tick_params(axis='y',labelsize=ls-2)
#ax3.legend(fontsize=bs,loc=9, ncol=3)
ax4.grid()
    
#fig.tight_layout()
    
plt.show()

