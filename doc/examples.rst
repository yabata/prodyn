.. currentmodule:: pyrenn

.. _examples:

Examples
============

The examples given in this chapter show how to implement dynamic programming algorithm. Here only brief description of each of the system is written. Very detailed comments through all codes will help to achieve deeper and detailed understanding. Results of optimal system control can be seen after simulations of the **system_example** codes.    

Building
^^^^^^^^
A system in **building** example contains a model of the real building (pre-trained Neural Network) and a heat pump. The goal of the optimization is to keep room temperature **Troom** inside the range of allowed values **[Tmin; Tmax]** in a cost-efficient way.
Simulation covers one day (19-44 hours) with 15 min time resolution. The picture in the Figure 10 visualizes current system.

.. figure:: img/building.png
   :width: 90%
   :align: center
   
   Figure 10: Illustration of the **building** example
 
Building with Storage
^^^^^^^^
The presence of the **heat storage** makes this system different to the **building**. Due to this **building_with_storage** has 4 decisions and 2 states (the room temperature **Troom** and energy content of the storage **E**). The goal and period of simulation are identical to the **building** example. In the Figure 11 schematic picture the **building_with_storage** system is given.

By reason of long-time simulation the results are already given in the folder related to this example. 

.. figure:: img/building_with_storage.png
   :width: 90%
   :align: center
   
   Figure 11: Illustration of the **building_with_storage** example
   
.. note::

	
	**Building** and **building_with_storage** examples can be simulated only in **forward** direction.  
   
CHP
^^^^^^^^
Grid, gas-boiler, chp power plant, battery and heat storage are components of the system, which should cover given heat and electical demand. Energy contents of the battery and heat storage are 2 states of the system. When **chp** is **on**, it covers the demand. Surplus of electricity is stored in the battery and sold to the grid. Surplus of the heat is stored in the heat storage. When **chp** is **off**, at first both demands are covered by storages, then by the grid and gas boiler. The goal of optimization is to find the path, where both storages will be empty at the final timestep.     

.. figure:: img/chp.png
   :width: 90%
   :align: center
   
   Figure 12: Illustration of the **chp** example


