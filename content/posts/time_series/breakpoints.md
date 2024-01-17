+++
title = 'Breakpoints'
date = 2024-01-16T09:48:42-03:00
draft = true
+++

* A *breakpoint* is a structural change in the data, such as an anomaly, change in trend, ...
* techniques to identify breakpoints can be classified into two categories: 
	1. detection: detect one or more breakpoints
	2. test if a given point is a breakpoint
* Python libraries to detect breakpoints: rupture, jenkspy
* Decide for how many breakpoits we are looking for
* Ruptures provides six different models to detect breakpoints: Dynp, KernelCPD, Pelt, Binseg, BottomUp, Window.
	* Each model must be built, trained (through the fit() function) and then used for prediction. The predict() function receives as input the number of breakpoints (minus 1) to be identified. 
