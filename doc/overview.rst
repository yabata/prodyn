.. currentmodule:: prodyn

.. _overview:

Overview
============
An overview explains the basic procedure of the dynamic programming implementation in the random example. It also introduces files, which are involved in the implementation process, and clarifies main functions inside these files.

Simplified diagram of the process and connection between documents involved in it are shown in the Figure 1.    

.. figure:: img/basic_procedure.png
   :width: 70%
   :align: center
   
   Figure 1: Scheme of the dynamic programming implementation
   
According to the diagram any implemetation consists of four files. Three of them (example_data, example_model and run_example) are specified and should be created for each current example. The fourth one (prodyn) is autonomous and can be used with any example without modifications. 

.. _ref-to-subchapter:

Example_data
--------------------------------------
Example_data is a file in excel-form. It stores an information about a system, which operation should be controlled in an optimal way. All data is split between four sheets: **Time-Series**, **Constants**, **DP-States** and **DP-Decisions**. Each of this sheet is described below. 

Time-Series
^^^^^^^^
A series of values for parameters, which described the system, is shown here. Values are obtained at successive times and with equal intervals between them. Small part of **Time-Series** from ``chp_data`` is illustrated in the Figure 2.   

.. figure:: img/time_series.png
   :width: 70%
   :align: center
   
   Figure 2: Time-Series from ``chp`` example
   
.. csv-table:: Sheet **Time-Series** from ``chp`` example
       :header-rows: 1
       :stub-columns: 4
           
        Time,el_demand,heat_demand,el_cost,el_feed-in
        1,0.24,5.07,0.16,0.06
        2,0.71,3.55,0.15,0.06
        3,0.23,2.47,0.15,0.06
        4,1.06,2.43,0.14,0.06
        
Constants
^^^^^^^^ 
This sheet keeps all values for parameters, which doesn't change during any operation of the system. 

DP-States
^^^^^^^^
        
