# A more complex MAF configuration that uses python loops to configure things for every filter

# to run:
# runDriver.py largeCfg.py

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Set the output directory
root.outputDir = './LargeCfg'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
# Name of the output table in the database
root.opsimName = 'opsimblitz2_1060'

# Make an empty list to hold all the slicer configs
sliceList = []

# Define the filters we want to loop over
filters = ['u','g','r','i','z','y']
# Make a dict of what colors to use for different filters
colors={'u':'m','g':'b','r':'g','i':'y','z':'Orange','y':'r'}

# Look at the single-visit depth and airmass for observations in each filter and merge
# them into a single histogram
for f in filters:
    m1 = configureMetric('CountMetric', kwargs={'col':'fiveSigmaDepth'}, 
                          histMerge={'histNum':1, 'legendloc':'upper right',
                                     'color':colors[f],'label':'%s'%f, 
                                     'ylabel':'Count'} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'fiveSigmaDepth','binsize':0.1,},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)]) 
    sliceList.append(slicer)
    m1 = configureMetric('CountMetric', kwargs={'col':'airmass'},
                          histMerge={'histNum':2, 'legendloc':'upper right',
                                     'color':colors[f], 'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'airmass','binsize':0.05},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)])
    sliceList.append(slicer)




root.slicers = makeDict(*sliceList)
