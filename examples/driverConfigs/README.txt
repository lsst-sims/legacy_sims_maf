This directory contains a variety of example MAF config scripts.

To run, set up your environment - 
'source loadLSST.csh' (or .sh),
'setup sims_maf', 

Then note that there are two different kinds of driver config files in
this directory: 'one-off' configuration files and 'flexible'
configuration files. 

Both are intended to be used with the sims_maf/bin/runDriver.py
command (after setting up your environment, this should be in your
path). 

One-off configuration files have all of their parameters hard coded 
	(i.e. the opsim sqlite database file and output directory are
	specified in the configuration file itself).

allSlicerCfg.py, f0Drive.py and maf_example_web.py
are all one-off configuration files. Some of these use the test sqlite
database (../../testes/opsimblitz1_1131_sqlite.db) and some are
intended to be used on a copy of opsimblitz2_1060 downloaded to this directory.

Flexible configuration files work with arguments passed from runDriver.py 
	     to set the opsim run name, sqlite database file location
	     and name, and output directory dynamically.

sstarDriver.py, cadenceDriver.py and install_initialtest.py are flexible driver configuration
files. They can be used on any sqlite database file by entering: 
       runDriver.py --runName RUNNAME --dbDir DBDIR --outputDir OUTDIR
       configfilename


Config file descriptions:
----------------
allSlicerCfg.py:  This file contains at least one example of every type of slicer available in MAF
maf_example_web.py:  Driver script for making example plots for Confluence documentation. 
fODrive.py: calculates f_O metric. (fO metric also calculated in sstarDriver.py).
install_initialtest.py: simple test script to check installation. 

cadenceDriver.py: This tests many of the new cadence metrics.
sstarDriver.py: Generates the plots and summary statistics currently
in SSTAR (plus a few additional metrics). 

