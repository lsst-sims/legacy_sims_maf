# Here is an example of a very very simple MAF configuration driver script
#  with lots of comments.
# To execute and run the MAF driver using this configuration script --
## runDriver.py very_simple_cfg.py

# This driver configuration script is a 'one-off' configuration script: the
#  opsim run name, database location and output directory are hard-coded into the file.

# This is a python file, and so we can use python in this script -- however, only lines
#  which start with 'root' are passed onto the MAF Driver.

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Set the output directory
root.outputDir = './Very_simple_out'
# Set the database to use the example db included in the git repo
#  Note that the dbAddress is made up of 'sqlite:///' and then the *relative* path
#  to the sqlite database file.
root.dbAddress = {'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
# Name of this opsim run -- this is used for plot titles and as metadata to identify the
#  run if we were to read the metric data back from disk after calculation.
root.opsimName = 'opsimblitz1.1131'

# Make an empty list to hold all the slicer configs
sliceList = []

# We want to run metrics in several filters individually: set a list of
#  SQL constraints consisting of 'where filter = u' (etc.).
# Note that in the SQL constraints passed to MAF we do not include 'where'. 
constraints = []
filters = ['u','g','r','i','z','y']
for f in filters:
    constraints.append("filter = '%s'"%f)
# Note that this list looks like the following -- 
#["filter = 'u'", "filter = 'g'", "filter = 'r'", "filter = 'i'", "filter = 'z'", "filter = 'y'"]

# Set up the metrics we want to use - here we'll 
#   run 2 metrics, the mean seeing and the co-added 5-sigma limiting depth.
metric1 = configureMetric('MeanMetric', params=['finSeeing'])
metric2 = configureMetric('Coaddm5Metric', plotDict={'cbarFormat':'%.3g'})

# Set up the slicer we want to use with these metrics -- 
#  we'll use the Healpix slicer (at a low resolution) to make sky maps and power spectra.
slicer = configureSlicer('HealpixSlicer', metricDict=makeDict(metric1, metric2),
                          kwargs={'nside':16}, constraints=constraints)
sliceList.append(slicer)


# Now we'll set up one more metric that we want to use with a UniSlicer.
metric = configureMetric('MeanMetric', params=['finSeeing'])
slicer = configureSlicer('UniSlicer', metricDict=makeDict(metric),
                          constraints=constraints)
sliceList.append(slicer)

# This is last step bundles up all of our configured metrics and slicers,
#   and pass the result into the MAF driver.
root.slicers = makeDict(*sliceList)

