.. currentmodule:: prodyn

.. _example_model:

Example_model
========================
**Example_model** is a file, which is written in python and should be created specifically for the current system. The main part of it is a function, which describes the transition of the system from one timestep to another one. Figure 5 gives simplified illustration of this transition.  

.. figure:: img/model.png
   :width: 80%
   :align: center
   
   Figure 5: System transition from timestep **i** to **j**
   
System's condition at timestep **i** is defined by an array **X**, which is built from the **DP-States** data in excel-form. The process of **X** formation is fully described in one of the next subchapters **prepare_DP**. Main function of the **Example_model** calculates the transition from **i** to **j** in dependance of each decision from the list of possible ones **U**. Results of the calculation are an array **Xj**, which describes the condition of the system at timestep **j**, and the **cost** of the transition for each possible decision from **U**.       





