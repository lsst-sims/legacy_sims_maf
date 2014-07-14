This directory contains a variety of example MAF config scripts.

To run, set up your environment - 
'source loadLSST.csh' (or .sh),
'setup sims_maf', 

Then note that there are two different kinds of driver config files in
this directory: 'one-off' configuration files and 'flexible'
configuration files. 

One-off configuration files are intended to be used with the
sims_maf/bin/runDriver.py command.

Flexible configuration files are intended to be used with the
sims_maf/bin/runFlexibleDriver.py command. If you try to run the wrong
kind of configuration file with the wrong 'run*Driver.py' you'll get
an error message directing you to the correct command.

----
One-off configuration files have all of their parameters hard coded 
	(i.e. the opsim sqlite database file and output directory are
	specified in the configuration file itself, and all config
	parameters start with 'root').

install_initialtest.py, allSlicerCfg.py, and maf_example_web.py
are all one-off configuration files. They are intended to be copied to
your working directory, and then
run on opsimblitz2_1060 present in the same directory (note that you
can edit the one-off configuration files in your working directory as needed).  
You can then run then using:  runDriver configfilename

----
Flexible configuration files work with arguments passed from runFlexibleDriver.py 
	     allowing the user to set the opsim run name, sqlite database file location
	     and name, and output directory at run-time.

install_initialtestFlexible.py, sstarDriver.py and cadenceDriver.py are flexible driver configuration
       files. They can be used on any sqlite database file by entering: 
       runFlexibleDriver.py --runName RUNNAME --dbDir DBDIR --outputDir OUTDIR
           configfilename
(note that you can also set the 'slicerName' .. see the driver
       configuration file itself for more information as to the
       available options). 

----------------
Config file descriptions:
----------------
install_initialtest.py: simple demo test script to check installation. 
allSlicerCfg.py:  This file contains at least one example of every type of slicer available in MAF
maf_example_web.py:  Driver script for making example plots for Confluence documentation. 

install_initialtestFlexible.py: simple demo test script to check
installation (this is the same as install_initialtest.py but in a
flexible format). 
sstarDriver.py: Generates the plots and summary statistics currently 
in SSTAR (plus a few additional metrics such as the fO metric). 
cadenceDriver.py: This tests many of the new cadence metrics.

