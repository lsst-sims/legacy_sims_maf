# Here is an example of a very very simple MAF configuration driver script
# to run:
# runDriver.py very_simple_cfg.py

# This script uses the LSST pex_config.  This is executed as a python script, but only things that start with
#'root.' are passed on to the driver script.

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Set the output directory
root.outputDir = './ManyMetrics'
# Set the database to use (the example db included in the git repo)
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
# Name of this run (filename base)
root.opsimName = 'opsimblitz2_1060'

# Make an empty list to hold all the slicer configs
sliceList = []

# Make a set of SQL where constraints to only use each filter
constraints = []
filters = ['u','g','r','i','z','y']
for f in filters:
    constraints.append("filter = '%s'"%f)
#["filter = 'u'", "filter = 'g'", "filter = 'r'", "filter = 'i'", "filter = 'z'", "filter = 'y'"]

# Run 2 metrics, the mean seeing and the co-added 5-sigma limiting depth.
metric1 = configureMetric('MeanMetric', kwargs={'col':'finSeeing'})
metric2 = configureMetric('Coaddm5Metric', plotDict={'cbarFormat':'%.3g'})

# Configure a slicer.  Use the Healpix slicer to make sky maps and power spectra.
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric1,metric2),
                          kwargs={'nside':16}, constraints=constraints)
sliceList.append(slicer)

metric = configureMetric('MeanMetric', kwargs={'col':'finSeeing'})
# Configure a slicer.  Use the UniSlicer to simply take all the data.
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric),
                          constraints=constraints)
sliceList.append(slicer)



root.slicers = makeDict(*sliceList)

