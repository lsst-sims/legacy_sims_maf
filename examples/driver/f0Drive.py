# Example of f_0 metric and driver.
# To run:
# runDriver.py f0Drive.py

from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict


small = False # Use the small database included in the repo

if small:
    root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
    root.opsimNames = ['opsim_small']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?
else:
    root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
    root.opsimNames = ['opsim']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?



root.outputDir = './f0out'
nside=128
leafsize = 50000 # For KD-tree

m1 = makeMetricConfig('CountMetric', params=['expMJD'], 
                      kwargs={'metricName':'f0'}, 
                      plotDict={'units':'Number of Visits', 'xMin':0, 
                                'xMax':1500},
                      summaryStats={'f0Area':{'nside':nside},
                                    'f0Nv':{'nside':nside}})
binner = makeBinnerConfig('f0Binner', kwargs={"nside":nside},
                          metricDict=makeDict(m1),
                          setupKwargs={"leafsize":leafsize},
                          constraints=['','propID = %s'%WFDpropid])

root.binners = makeDict(binner)
