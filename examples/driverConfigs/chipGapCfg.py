import numpy as np
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict


root.outputDir = './ChipGap'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz1_1133_sqlite.db'}
root.opsimName = 'opsimblitz1_1133'

root.verbose = True

slicerList=[]
constraints = ['filter="r"']
# How many Healpix sides to use
nside=16

m1 = configureMetric('CountMetric', kwargs={'col':'expMJD'},
                     plotDict={'percentileClip':80., 'units':'#'},
                     summaryStats={'MeanMetric':{},'RmsMetric':{}, 'CountMetric':{}})
m2 = configureMetric('Coaddm5Metric',
                     plotDict={'zp':27., 'percentileClip':95, 'units':'Co-add m5 - %.1f'%27.},
                     summaryStats={'MeanMetric':{},'RmsMetric':{})
# Combine metrics in a dictionary
metricDict = makeDict(m1,m2)
# Generate the slicer configuration, passing in the metric configurations and SQL constraints
slicer = configureSlicer('HealpixSlicer',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec"},
                          metricDict=metricDict, constraints=constraints)
# Add the slicer to the list of slicers
slicerList.append(slicer)

slicer = configureSlicer('HealpixSlicer',
                          kwargs={"nside":nside,'spatialkey1':"fieldRA", 'spatialkey2':"fieldDec", 'useCamera':True},
                          metricDict=metricDict, constraints=constraints, metadata='chipGaps')
# Add the slicer to the list of slicers
slicerList.append(slicer)


root.slicers=makeDict(*slicerList)
