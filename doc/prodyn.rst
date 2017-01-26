.. currentmodule:: prodyn

.. _prodyn:

Prodyn
============
**Prodyn** is an autonomous file, which is written in python and can be used with any example. This is the core and driving force of every dynamic programming implementation. Three main **prodyn's** functions are described below in details.       

.. _prepareDP-ref:

prepare DP 
^^^^^^^^
The goal of **prepare_DP** is a creation of several arrays, which will be used subsequently. The simplified procedure of the creation for 1 and 2 states random examples is presented in the Figure 6.

.. figure:: img/prepare_dp.png
   :width: 90%
   :align: center
   
   Figure 6: Working principle of **prepare_DP** function
   
The table with states of the system, which is stored in :ref:`DP-States <DP-States>` sheet of :ref:`system_data <system_data>`, plays a role of input for the **prepare_DP**. Three new arrays are the main returns of the function. **X** is an array containing every possible condition of the system. It's size depends on the number of system's states. For example, any  condition of the system with 2 states is always characterized by two variables and **X** is, consequently, 2d. In addition, every system's condition corresponds to the specific number. These numbers are stored in an array **Xidx**, which is always 1-d. The third return **XX** is an array of arrays, from which **X** is built. In other words, **X** is the cartesian product of **XX**.       


DP forward
^^^^^^^^
bla-bla

DP backward
^^^^^^^^
