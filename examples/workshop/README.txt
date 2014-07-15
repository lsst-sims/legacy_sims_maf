This folder contains example code used for the LSST Cadence Workshop:  https://project.lsst.org/meetings/ocw/

We expect users will run code in their own directories (to prevent work being overwritten in upgrades, etc).  For example, after setting up MAF, one might:
>cd
>mkdir MyMAF
>cd MyMAF
>cp $SIMS_MAF_DIR/examples/workshop/*.py .

The example configuration files point to OpSim simulations that must be downloaded.  
You can grab them with the following commands:
>curl -O http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1060/design/opsimblitz2_1060_sqlite.db
>curl -O http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1056/design/opsimblitz2_1056_sqlite.db

The files can then be run as, e.g.:
>runDriver.py oneMetricOneSlicer.py

