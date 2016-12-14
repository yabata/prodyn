import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrenn as prn
import mpl_settings as ms
ms.set_style()

file = 'building_data.xlsx'
Srs = pd.read_excel(file,sheetname='Time-Series',index_col=[0])
net = prn.loadNN('NN_building.csv')

Srs['massflow'] = 200
Srs['T_room'] = 20
Srs['P_th'] = 1

delay = 4

timesteps = np.arange(4,287)

for t in timesteps:
	
	hour0 = Srs.loc[t-delay:t-1]['hour'].values.copy()
	solar0 = Srs.loc[t-delay:t-1]['solar'].values.copy()
	T_amb0 = Srs.loc[t-delay:t-1]['T_amb'].values.copy()
	user0  = Srs.loc[t-delay:t-1]['use_room'].values.copy()
	T_inlet0 = Srs.loc[t-delay:t-1]['T_inlet'].values.copy()
	massflow0 = Srs.loc[t-delay:t-1]['massflow'].values.copy()
	
	T_room0 = Srs.loc[t-delay:t-1]['T_room'].values.copy()
	P_th0 = Srs.loc[t-delay:t-1]['P_th'].values.copy()

	hour = Srs.loc[t]['hour']
	solar = Srs.loc[t]['solar']
	T_amb = Srs.loc[t]['T_amb']
	user  = Srs.loc[t]['use_room']
	T_inlet = Srs.loc[t]['T_inlet']
	massflow = Srs.loc[t]['massflow']

	P0 = np.array([hour0,solar0,T_amb0,user0,massflow0,T_inlet0],dtype = np.float)
	Y0 = np.array([T_room0,P_th0],dtype = np.float)
	P = np.array([[hour],[solar],[T_amb],[user],[massflow],[T_inlet]],dtype = np.float)

	T_room,P_th = prn.NNOut(P,net,P0=P0,Y0=Y0)
	
	Srs.loc[t,'P_th'] = P_th
	Srs.loc[t,'T_room'] = T_room
	
fig = plt.figure(figsize=[11,7])
ax0 = fig.add_subplot(211)
ax0.plot(Srs.loc[timesteps]['T_room'])

ax1 = fig.add_subplot(212)
ax1.plot(Srs.loc[timesteps]['P_th'])

plt.show()