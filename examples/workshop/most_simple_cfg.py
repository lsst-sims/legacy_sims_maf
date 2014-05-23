# Here is an example of a very very simple MAF configuration driver script
# to run:
# runDriver.py most_simple_cfg.py

# This script uses the LSST pex_config.  This is executed as a python script, but only things that start with 'root.' are passed on to the driver script.

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict

# Set the output directory
root.outputDir = './Most_simple_out'
# Set the database to use (the example db included in the git repo)
root.dbAddress = {'dbAddress':'sqlite:///../opsim_small.sqlite'}
# Name of the output table in the database
root.opsimNames = ['opsim_small']

# Configure a metric to run. Compute the mean on the final delivered seeing.  Use the IdentityMetric as a summary stat to pass the result to the summaryStats file.
metric = makeMetricConfig('MeanMetric', params=['finSeeing'],
                          summaryStats={'IdentityMetric':{}})

# Configure a binner.  Use the UniBinner to simply take all the data.
binner = makeBinnerConfig('UniBinner', metricDict=makeDict(metric),
                          constraints=[''])

root.binners = makeDict(binner)

