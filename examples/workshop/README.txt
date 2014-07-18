This folder contains example code used for the LSST Cadence Workshop:  https://project.lsst.org/meetings/ocw/

We expect users will run code in their own directories (to prevent work being overwritten in upgrades, etc).  For example, after setting up MAF, one might:
>cd
>mkdir MyMAF
>cd MyMAF
>cp $SIMS_MAF_DIR/examples/workshop/*.py .

The example configuration files point to OpSim simulations that must be downloaded.  
You can grab them with the following commands:
>curl -O http://www.noao.edu/lsst/opsim/CadenceWorkshop2014/opsimblitz2_1060_sqlite.db
>curl -O http://www.noao.edu/lsst/opsim/CadenceWorkshop2014/opsimblitz2_1049_sqlite.db

The files can then be run as, e.g.:
>runDriver.py oneMetricOneSlicer.py

