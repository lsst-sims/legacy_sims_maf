# Example of f_O metric and driver.
# To run:
# runDriver.py fODrive.py

from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


root.dbAddress = {'dbAddress':'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
# Connect to the database to fetch some values we're using to help configure the driver.                                                             
opsimdb = utils.connectOpsimDb(root.dbAddress)
# Fetch the proposal ID values from the database
propids, WFDpropid, DDpropid = opsimdb.fetchPropIDs()

# Construct a WFD SQL where clause so multiple propIDs can by WFD:                                                                                   
wfdWhere = ''
if len(WFDpropid) == 1:
    wfdWhere = "propID = '%s'"%WFDpropid[0]
else:
    for i,propid in enumerate(WFDpropid):
        if i == 0:
            wfdWhere = wfdWhere+'('+'propID = %s'%propid
        else:
            wfdWhere = wfdWhere+'or propID = %s'%propid
        wfdWhere = wfdWhere+')'

root.outputDir = './fOout'
nside=128
leafsize = 50000 # For KD-tree

m1 = configureMetric('CountMetric', params=['expMJD'], 
                      kwargs={'metricName':'fO'}, 
                      plotDict={'units':'Number of Visits', 'xMin':0, 
                                'xMax':1500},
                      summaryStats={'fOArea':{'nside':nside},
                                    'fONv':{'nside':nside}})
slicer = configureSlicer('fOSlicer', kwargs={"nside":nside},
                          metricDict=makeDict(m1),
                          setupKwargs={"leafsize":leafsize},
                          constraints=['',wfdWhere])

root.slicers = makeDict(slicer)
