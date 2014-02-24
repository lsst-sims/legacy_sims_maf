# A MAF config that replicates the SSTAR plots

import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './Plots'
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
    binner = BinnerConfig()
    binner.name = 'HealpixBinner'
    binner.kwargs = {"nside":nside}
    m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'percentileClip':80., 'units':'#'})
    m2 = makeMetricConfig('CountRatioMetric', params=['expMJD'], kwargs={'normVal':nvisitBench[i]},plotDict={'percentileClip':80.})
    m3 = makeMetricConfig('MedianMetric', params=['5sigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[i], 'percentileClip':True, 'plotLabel':'Co-add m5 - %.1f'%mag_zpoints[i]} )             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[i]})
    m6 = makeMetricConfig('MedianMetric', params=['seeing'], plotDict={'normVal':seeing_norm[i], 'plotLabel':'median seeing/expected zenith seeing'})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'])
    m8 = makeMetricConfig('MaxMetric', params=['airmass'])
    binner.metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner.setupKwargs_float={ "leafsize":50000}
    binner.setupParams=["fieldRA","fieldDec"]
    binner.constraints = ["filter = \'%s\' and propID = 218"%f, "filter = \'%s\'"%f]
    binList.append(binner)


# Visits per observing mode:
modes = [214,215,216,217,218,219]
for i,f in enumerate(filters):
        binner = BinnerConfig()
        binner.name='HealpixBinner'
        binner.kwargs = {"nside":nside}
        m1 = makeMetricConfig('CountMetric', params=['expMJD'],plotDict={'percentileClip':95., 'units':'#'})
        binner.metricDict = makeDict(m1)
        binner.setupKwargs={ "leafsize":50000}
        binner.setupParams=["fieldRA","fieldDec"]
        for mode in modes:
            binner.constraints.append("filter = \'%s\' and propID = %s"%(f,mode))
        binList.append(binner)
                                    
        

        
# Slew histograms
binner= BinnerConfig()
binner.name='OneDBinner'
binner.setupParams=['slewTime']
m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner.metricDict=makeDict(m1)
binner.constraints=['']
binList.append(binner)

binner= BinnerConfig()
binner.name='OneDBinner'
binner.setupParams=['slewDist']
m1 = makeMetricConfig('CountMetric', params=['slewDist'],kwargs={'metadata':'dist'} )
binner.metricDict=makeDict(m1)
binner.constraints=['']
binList.append(binner)

# Compute what fraction of possible observing time the shutter is open
binner= BinnerConfig()
binner.name='UniBinner'
m1 = makeMetricConfig('SummaryStatsMetric')
binner.metricDict=makeDict(m1)
binner.constraints=['night < 49353+730', '']
binList.append(binner)



root.binners=makeDict(*binList)


