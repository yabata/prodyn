import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

import chp_model as model
import prodyn as prd

file = 'chp_data.xlsx'

#define constants, timeseries, list of possible decisions U and
#timesteps, which are needed for the model
cst,srs,U,states = model.read_data(file)
timesteps=np.arange(1,49)


#define function for simulation that calculates costs and next state
system=model.chp

#optimize with DP forward algorithm
result = prd.DP_backward(states,U,timesteps,cst,srs,system,verbose=True,t_verbose=5)

#get result with empty storages at the end (index=0)
result0 = result.xs(0,level='Xidx_start')


############################################################################
############################################################################
#access results
bat_pow = result0.loc[timesteps]['bat_pow'].values.astype(np.float)
heatsto_pow = result0.loc[timesteps]['heatsto_pow'].values.astype(np.float)
el_demand_ext = result0.loc[timesteps]['el_demand_ext'].values.astype(np.float)
boiler = result0.loc[timesteps]['heat_demand_boiler'].values.astype(np.float)

battery = result0['battery']
heatsto = result0['heat-storage']

el_demand = srs.loc[timesteps]['el_demand'].values
heat_demand = srs.loc[timesteps]['heat_demand'].values

opt_control = result0.loc[timesteps]['U']

#######################################
#Prepare results for plotting
grid_import = np.zeros(len(timesteps))
grid_import[el_demand_ext>0] = el_demand_ext[el_demand_ext>0]

grid_export = np.zeros(len(timesteps))
grid_export[el_demand_ext<0] = el_demand_ext[el_demand_ext<0]

bat_charge = np.zeros(len(timesteps))
bat_charge[bat_pow>0] = bat_pow[bat_pow>0]

bat_discharge = np.zeros(len(timesteps))
bat_discharge[bat_pow<0] = bat_pow[bat_pow<0]

heat_charge = np.zeros(len(timesteps))
heat_charge[heatsto_pow>0] = heatsto_pow[heatsto_pow>0]

heat_discharge = np.zeros(len(timesteps))
heat_discharge[heatsto_pow<0] = heatsto_pow[heatsto_pow<0]

chp_el = np.zeros(len(timesteps))
chp_el[opt_control.values=='on'] = cst['chp_Pmax_el']

chp_th = np.zeros(len(timesteps))
chp_th[opt_control.values=='on'] = cst['chp_Pmax_el']/cst['chp_eff_el']*cst['chp_eff_th']

####################################
#plot
#####
fig = plt.figure(figsize=(12,8))
gs = mpl.gridspec.GridSpec(3, 1, height_ratios=[1, 1,0.7])
fs = 18
ms=12

##Subplot 1
ax1 = fig.add_subplot(gs[0])
ax1.set_title('Electricity',fontsize=fs)
ax1.plot(timesteps,el_demand,label='demand',lw=3,color='k')
ax1.stackplot(timesteps,chp_el,grid_import,-bat_discharge,\
				colors = ['salmon','cornflowerblue','lightgreen'],lw=0)
ax1.stackplot(timesteps,grid_export,-bat_charge,\
				colors = ['navy','green'],lw=0)

ax1.grid()
ax1.tick_params(labelsize=fs-2)
ax1.set_ylabel('Power [kW]',fontsize = fs)
plt.setp(ax1.get_xticklabels(), visible=False)
				
#Legend
ax1.plot([],marker='s',color ='salmon',label='chp',linestyle = 'None',ms = ms)
ax1.plot([],marker='s',color ='cornflowerblue',label='grid import',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='navy',label='grid export',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='lightgreen',label='from storage',linestyle = 'None',ms = ms)	
ax1.plot([],marker='s',color ='green',label='into storage',linestyle = 'None',ms = ms)				
ax1.legend(fontsize=fs-4,numpoints = 1,handlelength=1)

##Subplot 2
ax2 = fig.add_subplot(gs[1],sharex=ax1)
ax2.set_title('Heat',fontsize=fs)
ax2.plot(timesteps,heat_demand,label='demand',lw=3,color='k')
ax2.stackplot(timesteps,chp_th,boiler,-heat_discharge,\
				colors = ['salmon','grey','lightgreen'],lw=0)
ax2.stackplot(timesteps,-heat_charge,\
				colors = ['green'],lw=0)

ax2.grid()
ax2.tick_params(labelsize=fs-2)
ax2.set_ylabel('Power [kW]',fontsize = fs)
plt.setp(ax2.get_xticklabels(), visible=False)

				
#Legend
ax2.plot([],marker='s',color ='salmon',label='chp',linestyle = 'None',ms = ms)
ax2.plot([],marker='s',color ='grey',label='boiler',linestyle = 'None',ms = ms)
ax2.plot([],marker='s',color ='lightgreen',label='from storage',linestyle = 'None',ms = ms)	
ax2.plot([],marker='s',color ='green',label='into storage',linestyle = 'None',ms = ms)		
ax2.legend(fontsize=fs-4,numpoints = 1,handlelength=1)



##Subplot 3
ax3 = fig.add_subplot(gs[2],sharex=ax1)
ax3.set_title('Energy content',fontsize=fs)
ax3.plot(battery,lw=3,color='cornflowerblue', label = 'battery')
ax3.plot(heatsto,lw=3,color='green', label = 'heat storage')


ax3.grid()
ax3.tick_params(labelsize=fs-2)
ax3.set_ylabel('[kWh]',fontsize = fs)
ax3.set_xlabel('time [h]',fontsize = fs)
ax3.set_xlim((0,62))

ax3.legend(fontsize=fs-4,numpoints = 1,handlelength=1, loc='upper right')

plt.show()
