.. prodyn documentation master file, created by
   sphinx-quickstart on Mon Jan 11 11:43:04 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: prodyn

prodyn: Implementation of the dymaic programming algorithm for optimal system control
================================================================

:Maintainer: Dennis Atabay, <dennis.atabay@tum.de>
:Organization: `Institute for Energy Economy and Application Technology`_,
               Technische Universität München
:Version: |version|
:Date: |today|
:Copyright:
  This documentation is licensed under a `Creative Commons Attribution 4.0 
  International`__ license.

.. __: http://creativecommons.org/licenses/by/4.0/



Contents
--------

This documentation contains the following pages:

.. toctree::
   :maxdepth: 1

   overview
   example_data
   example_model
   prodyn
   run_example
   examples
   

Features
--------
* mutiple states...


Get Started
-----------

1. `download`_ or clone (with `git`_) this repository to a directory of your choice.
2. Copy the ``prodyn.py`` file in the ``prodyn`` folder to a directory which is already in python's search path or add the ``prodyn`` folder to python's search path (sys.path) (`how to`__)
3. Run the given examples in the `examples` folder.
4. Implement your own system function.


.. __: http://stackoverflow.com/questions/17806673/where-shall-i-put-my-self-written-python-packages/17811151#17811151) 
.. _how to: http://www.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html


  
Dependencies (Python)
------------

* `numpy`_ for mathematical operations 
* `pandas`_ only for using the examples 


   
.. _Institute for Energy Economy and Application Technology: http://www.ewk.ei.tum.de/
.. _numpy: http://www.numpy.org/
.. _pandas: https://pandas.pydata.org
.. _download: https://github.com/yabata/prodyn/archive/master.zip
.. _git: http://git-scm.com/
