.. currentmodule:: prodyn

.. _example_model:

Example_model
========================
Example-model is a file, which is written in python and should be created specifically for the current system. The main part of it is a function, which describes the transition of the system from one timestep to another one. Figure 5 gives simplified illustration of this transition.  

.. figure:: img/model.png
   :width: 70%
   :align: center
   
   Figure 5: System transition from timestep **i** to **j**
   
System's condition at timestep **i** is defined by an array **X**, which is built from the **DP-States** data in excel-form. The process of **X** formation is fully described in one the next subchapters **Prodyn**. Similarly the condition at **j** is described by **Xj**. Main function of the **example_model** calculates the transition from **i** to **j** in dependance of each desicion from the list of possible ones **U**.    





