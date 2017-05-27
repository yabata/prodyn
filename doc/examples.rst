.. currentmodule:: pyrenn

.. _examples:

Examples
============

The examples given in this chapter show how to implement dynamic programming algorithm. Here only brief description of each of the system is written. Very detailed comments through all codes will help to achieve deeper and detailed understanding. Results of optimal system control can be seen after simulations of the **run_system** codes.    

Building
^^^^^^^^

Description
"""""""""""
A system in **building** example contains a model of the real building (pre-trained Neural Network) and a heat pump. The goal of the optimization is to keep room temperature **Troom** inside the range of allowed values **[Tmin; Tmax]** in a cost-efficient way.
Simulation covers one day (19-44 hours) with 15 min time resolution. The picture in the Figure 11 visualizes current system.

.. figure:: img/building.png
   :width: 90%
   :align: center
   
   Figure 11: Illustration of the **building** example
   
Dynamic Programming algorithm for optimal control of the **building** is realized with using four following files: 
	
	* **building_data.xlsx** - stores information about the system.
   	* **building_model.py** - reads system's data and describes transition from one timestep to another.
   	* **prodyn.py** - realizes dynamic programming algorithm.
   	* **run_building_forward.py** - runs the simulation and finds the optimal system's control.
	
**Run_building_forward.py** and **building_model.py** are described in detail below. 
	

   
run_building_forward.py
"""""""""""""""""""""""
Here is the script of the ``run_building_forward.py`` (one from four dynamic programming files) is explained step by step for better understanding. 

::

    import numpy as np
    import matplotlib.pyplot as plt
    import pyrenn as prn
    
Three packages are included: 

* `numpy`_ is the fundamental package for scientific computing with Python;
* `matplotlib.pyplot`_ is a plotting library which allows present results in a diagram form quite easily;
* `pyrenn`_ is a recurrent neural network toolbox for Python. 

::

	import building_model as model
	import prodyn as prd
	
Then **building_model** and **prodyn** (two other files of dynamic programming) are imported. They assigned as **model** and **prd** respectively. 

::

	file = 'building_data.xlsx'
	
Gives the path to the excel-file **building_data** containing data about the current system. This is the last file of dynamic programming.  

::

	cst,srs,U,states = model.read_data(file)
	srs['massflow'] = 0
	srs['P_th'] = 0
	srs['T_room'] = 20
	
Defines constants **cst**, timeseries **srs**, list of possible decisions **U** and parameters **states**, which characterize each possible **building's** state, by reading the **building_data** file. Process of reading is realized due to **read_data** function hidden in the **building_model** (model) file. To timeseries **srs** written from **building_data** some extra data is added. 

::

	timesteps=np.arange(cst['t_start'],cst['t_end'])
	
Sets a timeframe on which optimization will be realized. 

::

	net = prn.loadNN('NN_building.csv') 
	cst['net'] = net
	
Defines a model **net** of the real building (pre-trained Neural Network) and saves it to the constants **cst**. 

::

	xsteps=np.prod(states['xsteps'].values)
	J0 = np.zeros(xsteps)
	idx = prd.find_index(np.array([20]),states)
	J0[idx] = -9999.9
	
Creates an array **J0** of initial terminal costs. **J0** will be changed from transition to transition according to list of possible decisions **U** and will keep all costs. Due to stored infromation in **J0** optimal control of the **building** can be found. 

::

	idx = prd.find_index(np.array([20]),states)
	J0[idx] = -9999.9
	
Shifts the initial postition to index with temperature equaled to 20 degrees. 

::

	system=model.building
	
Defines function **building** from **building_model** for characterization the transition from one timestep to another.

::

	result = prd.DP_forward(states,U,timesteps,cst,srs,system,J0=J0,verbose=True,t_verbose=5)
	i_mincost = result.loc[cst['t_end']-1]['J'].idxmin()
	opt_result = result.xs(i_mincost,level='Xidx_end')
	
Implements dynamic programming algorithm for the chosen timeframe and saves all data to the **result**. Then finds index for cost-minimal path, extracts it from **result** and saves to **opt_result**. 

::

	best_massflow=opt_result['massflow'].values[:-1]
	Troom=opt_result['T_room'].values[:-1]
	Pel=opt_result['P_el'].values[:-1]

Choses parameters, which characterize cost-efficient **building** control system, and extracts them from **opt_result**. **Best_massflow** is a schedule, which shows at which timestep heat pump is switched on and at which switched off. **Pel** defines consumed electrical power, **Troom** - room temperature inside the house, which shouldn't be out of the comfort zone **[Tmin; Tmax]**.   

::

	Troom=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['T_room'],Troom))
	Pel=np.concatenate((srs.loc[timesteps[0]-4:timesteps[0]-1]['P_th'],Pel))
	
Sums values for timesteps, which were not involved in the optimization, with those, which were extracted from **opt_result**. The remaining part of the code is responsible for plotting chosen and additional parameters. They are presented in the Figure 12. 

.. figure:: img/building_results.png
   :width: 100%
   :align: center
   
   Figure 12: Cost-minimal control of the **building** for keeping **Troom** inside **[Tmin; Tmax]**.  


building.model.py
"""""""""""""""""
bla-bla

Building with Storage
^^^^^^^^
The presence of the **heat storage** makes this system different to the **building**. Due to this **building_with_storage** has 4 decisions and 2 states (the room temperature **Troom** and energy content of the storage **E**). The goal and period of simulation are identical to the **building** example. In the Figure 13 schematic picture the **building_with_storage** system is given.

By reason of long-time simulation the results are already given in the folder related to this example. 

.. figure:: img/building_with_storage.png
   :width: 90%
   :align: center
   
   Figure 13: Illustration of the **building_with_storage** example
   
.. note::

	
	**Building** and **building_with_storage** examples can be simulated only in **forward** direction.  
   
CHP
^^^^^^^^
Grid, gas-boiler, chp power plant, battery and heat storage are components of the system, which should cover given heat and electical demand. Energy contents of the battery and heat storage are 2 states of the system. When **chp** is **on**, it covers the demand. Surplus of electricity is stored in the battery and sold to the grid. Surplus of the heat is stored in the heat storage. When **chp** is **off**, at first both demands are covered by storages, then by the grid and gas boiler. The goal of optimization is to find the path, where both storages will be empty at the final timestep. Figure 14 shows simplified scheme of the **chp** system.     

.. figure:: img/chp.png
   :width: 90%
   :align: center
   
   Figure 14: Illustration of the **chp** example
   
PV Storage
^^^^^^^^
Photovoltaic system with storage form the system for covering given electrical demand. Energy content of the storage is the only state of the system. List **U** contains three possible decisions. With **normal** system operates without participation of the storage. Possible surplus of the produced by pv power can be saved in the storage with **charge** decision. With **discharge** system tries to cover the residual demand by stored energy. After each possible system's decision **grid load** is checked. This residual power is covered by or fed into the grid. The main goal is to find the result, where the storage is empty at the end. Illustration of the current example is presented in the Figure 15.       

.. figure:: img/pv_storage.png
   :width: 90%
   :align: center
   
   Figure 15: Illustration of the **pv_storage** example
   
**Pv_storage_model**, which describes the transition from **i** to **j** according to each possible decision **u**, is written in two ways. In first case the transition is applied for the whole **array X**, which characterizes the system. In the second case - for each possible condition of **X**. Calculation for each condition and jump from one to another are realized inside the **loop**.     



.. _numpy: https://docs.scipy.org/doc/numpy-dev/user/index.html
.. _matplotlib.pyplot: https://matplotlib.org/index.html
.. _pyrenn: https://github.com/yabata/pyrenn
