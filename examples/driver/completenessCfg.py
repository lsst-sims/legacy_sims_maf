# A MAF config to measure joint completeness

import numpy as np
from lsst.sims.operations.maf.driver.mafConfig import *

# Setup Database access
root.outputDir = './Complete'
root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
root.opsimNames = ['opsim']

filters = ['u','g','r','i','z','y']
#filters=['r']

# 10 year Design Specs
nvisitBench=[56,80,184,184,160,160] 
mag_zpoints=[26.1,27.4,27.5,26.8,26.1,24.9] 
sky_zpoints = [21.8,22.,21.3,20.0,19.1,17.5]
seeing_norm = [0.77,0.73,0.7,0.67,0.65,0.63]

binList=[]


# Completeness and Joint Completeness
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)','units':'# visits / # WFD','plotMin':.5, 'plotMax':1.5}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats=['mean'])

# For just WFD proposals
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=makeDict(m1), metadata='WFD', constraints=["propID = 188"])
binList.append(binner)

# For all Observations
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (all) / (# WFD Requested)','units':'# visits / # WFD','plotMin':.5, 'plotMax':1.5}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats=['mean'])
binner = makeBinnerConfig('OpsimFieldBinner',metricDict=makeDict(m1),constraints=[""])
binList.append(binner)




root.binners=makeDict(*binList)


