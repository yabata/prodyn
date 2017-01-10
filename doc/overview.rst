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
   
   Figure 2: **Time-Series** from ``chp`` example
        
Constants
^^^^^^^^ 
This sheet keeps all values for parameters, which doesn't change during any operation of the system. 

DP-States
^^^^^^^^
The part of the system, which operation should be optimized, is characterized by a number of states. Each state has min, max allowable values and number of steps between them. All these data is stored in **DP-States** sheet. **DP-States** for ``chp`` example is shown in the Figure 3.   

.. figure:: img/DP_states.png
   :width: 60%
   :align: center
   
   Figure 3: **DP-States** from ``chp`` example
   
As seen from the Figure 3 the system has two states. State of the **battery** can take values from **0** to **5** with a step equaled to **0,1**. Similarly **heat-storage** is changing between **0** and **10** with a step **0,1**. 

DP-Decisions
^^^^^^^^
An operation of the system for every timestep can be influenced by one of the specific decisions, which are written in **DP-Decisions** sheet. In other words, all possibilities for system control are written here. Figure 4 illustrates decisions for the same ``chp`` example.  

.. figure:: img/DP_decisions.png
   :width: 50%
   :align: center
   
   Figure 4: **DP-Decisions** from ``chp`` example
   
As seen above ``chp`` example has only two decisions: **off** and **on** operation of the combined heat and power plant. 

.. _ref-to-subchapter:

Example_model
--------------------------------------
Example-model is a file, which is written in python and should be created specifically for the current system. The main part of it is a function, which describes the transition of the system from one timestep to another one. Figure 5 gives simplified illustration of this transition.  

.. figure:: img/model.png
   :width: 70%
   :align: center
   
   Figure 5: System transition from timestep **i** to **j**
   
System's condition at timestep **i** is defined by an array **X**, which is built from the **DP-States** data in excel-form. The process of **X** formation is fully described in one the next subchapters **Prodyn**. Similarly the condition at **j** is described by **Xj**. Main function of the **example_model** calculates the transition from **i** to **j** in dependance of each desicion from the list of possible ones **U**.    


















