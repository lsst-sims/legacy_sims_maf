# A MAF config that replicates the SSTAR plots

import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './Plots_SD'
root.dbAddress ='sqlite:///opsim.sqlite'
root.opsimNames = ['opsim']

filters = ['u','g','r','i','z','y']
#filters=['r']

# 10 year Design Specs
nvisitBench=[56,80,184,184,160,160] 
mag_zpoints=[26.1,27.4,27.5,26.8,26.1,24.9] 
sky_zpoints = [21.8,22.,21.3,20.0,19.1,17.5]
seeing_norm = [0.77,0.73,0.7,0.67,0.65,0.63]

binList=[]
# Healpix resolution
nside = 128

for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'title': 'filter = %s'%f, 'percentileClip':75., 'units':'#'})
    m2 = makeMetricConfig('CountRatioMetric', params=['expMJD'], kwargs={'normVal':nvisitBench[i]},plotDict={'title': 'filter = %s'%f, 'percentileClip':80.})
    m3 = makeMetricConfig('MedianMetric', params=['5sigma_modified'],plotDict={'title': 'filter = %s'%f})
    m4 = makeMetricConfig('Coaddm5Metric', plotDict={'title': 'filter = %s'%f,'zp':mag_zpoints[i], 'percentileClip':95., 'units':'Co-add m5 - %.1f'%mag_zpoints[i]} )             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'title': 'filter = %s'%f, 'zp':sky_zpoints[i]})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'title': 'filter = %s'%f, 'normVal':seeing_norm[i], 'units':'median seeing/expected zenith seeing'})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'],plotDict={'title': 'filter = %s'%f})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'],plotDict={'title': 'filter = %s'%f} )
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\' and propID = 188"%f, "filter = \'%s\'"%f])
    binList.append(binner)


# Visits per observing mode:
modes = [186,187,188,189,190]
for i,f in enumerate(filters):
        m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'title': 'filter = %s'%f, 'units':'#'})
        metricDict = makeDict(m1)
        constraints=[]
        for mode in modes:
            constraints.append("filter = \'%s\' and propID = %s"%(f,mode))
        binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints )
        binList.append(binner)
                                    
        
# Slew histograms
m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
binList.append(binner)

m1 = makeMetricConfig('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
binList.append(binner)


# Compute what fraction of possible observing time the shutter is open
m1 = makeMetricConfig('SummaryStatsMetric')
binner = makeBinnerConfig('UniBinner', metricDict=makeDict(m1), constraints=['night < 730', ''])
binList.append(binner)



root.binners=makeDict(*binList)


