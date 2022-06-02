# GEO3000Code
Code for the project in GEO3000.

Here the program used for baseflow separation is listen. 
The program is ran as follows:

... main.py 'Mode' 'Method' 'Parameter'

Where 'Mode' signifies the run to be done.
For new data, this should be done using the keyword 'Clean', followed by all subsequent runs using 'Base'.

'Method' is either 'lh' or 'wf', for Lyne and Hollick or smoothed minima technique.

Parameter is the parameter used in either lh or wf, with wf standard being 0.9. 
