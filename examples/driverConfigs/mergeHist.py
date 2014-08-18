import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.outputDir = './MergeHist'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName = 'opsimblitz2_1060'
slicerList=[]


root.figformat = 'png'
filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}

slicerName='HealpixSlicer'
slicerkwargs={'nside':32}

for f in filters:
    m4 = configureMetric('Coaddm5Metric', plotDict={},
                         summaryStats={'MeanMetric':{}, 'RmsMetric':{}},
                         histMerge={'histNum':6, 'binsize':0.2, 'legendloc':'upper right',
                                    'color':colors[f], 'label':'%s'%f})
    metricDict = makeDict(m4)
    constraints = ['filter = "%s"' %(f)]
    slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                             constraints=constraints)
    slicerList.append(slicer)


for f in filters:
    m1 = configureMetric('CountMetric', kwargs={'col':'fiveSigmaDepth'},
                         histMerge={'histNum':1, 'legendloc':'upper right',
                                    'color':colors[f], 'label':'%s'%f} )

    slicer = configureSlicer('OneDSlicer',
                             kwargs={'sliceColName':'fiveSigmaDepth', 'binsize':0.1},metricDict=makeDict(m1),
                             constraints=["filter = '%s'"%(f)]) 
    #slicerList.append(slicer)


root.slicers=makeDict(*slicerList)
