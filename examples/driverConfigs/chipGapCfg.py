import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.outputDir = './ChipGap'
root.dbAddress = {'dbAddress':'sqlite:///ops1_1140_sqlite.db'}
root.opsimName = 'ops1_1140'
#root.dbAddress = {'dbAddress':'sqlite:///opsimblitz1_1133_sqlite.db'}
#root.opsimName = 'opsimblitz1_1133'

root.verbose = True

slicerList=[]
constraints = ['filter="r"']
#constraints = ['filter="r" and night < 100']
# How many Healpix sides to use
nside=128

m1 = configureMetric('CountMetric', kwargs={'col':'expMJD'},
                     plotDict={'percentileClip':80., 'units':'#'},
                     summaryStats={'MeanMetric':{},'RmsMetric':{}, 'SumMetric':{}})
m2 = configureMetric('Coaddm5Metric',
                     plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.},
                     summaryStats={'MeanMetric':{},'RmsMetric':{}})
# Combine metrics in a dictionary
metricDict = makeDict(m1,m2)
# Generate the slicer configuration, passing in the metric configurations and SQL constraints
slicer = configureSlicer('HealpixSlicer',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict=metricDict, constraints=constraints)
# Add the slicer to the list of slicers
slicerList.append(slicer)

slicer = configureSlicer('HealpixSlicer',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec",
                                  'useCamera':True, 'verbose':True},
                          metricDict=metricDict, constraints=constraints, metadata='chipGaps')
# Add the slicer to the list of slicers
slicerList.append(slicer)


root.slicers=makeDict(*slicerList)
