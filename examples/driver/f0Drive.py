# Example of f_0 metric and driver.
# To run:
# runDriver.py f0Drive.py

from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict


small = True # Use the small database included in the repo

if small:
    root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
    root.opsimNames = ['opsim_small']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?
else:
    root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
    root.opsimNames = ['opsim']
    propids = [215, 216, 217, 218, 219]
    WFDpropid = 217
    DDpropid = 219



root.outputDir = './f0out'
nside=128
leafsize = 50000 # For KD-tree

m1 = makeMetricConfig('CountMetric', params=['expMJD'], 
                      kwargs={'MetricName':'Nvisits'}, 
                      plotDict={'units':'Number of Visits'},
                      summaryStats={'f0Area':{},'f0Nv':{}})
binner = makeBinnerConfig('f0Binner', kwargs={"nside":nside},
                          metricDict=makeDict(m1),
                          setupKwargs={"leafsize":leafsize},
                          constraints=['','propID = %s'%WFDpropid])

root.binners = makeDict(binner)
