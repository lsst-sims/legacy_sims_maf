This folder contains example code used for the LSST Cadence Workshop:  https://project.lsst.org/meetings/ocw/

The example configuration files point to OpSim simulations that must be downloaded.  You can grab them with the following commands:

>curl -O http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1039/design/opsimblitz2_1039_sqlite.db
>curl -O http://opsimcvs.tuc.noao.edu/runs/opsimblitz2.1040/design/opsimblitz2_1040_sqlite.db

The files can then be run as, e.g.:
>runDriver.py most_simple_cfg.py

