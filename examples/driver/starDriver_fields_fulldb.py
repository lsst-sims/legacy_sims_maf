# A MAF config that replicates the SSTAR plots

import numpy as np
from lsst.sims.maf.driver.mafConfig import *
from lsst.sims.maf.utils import runInfo

# Setup Database access
root.outputDir ='./StarOut_Fields_full'
root.dbAddress ={'dbAddress':'sqlite:///opsim_hewelhog_1016.sqlite', 'fieldTable':'Field', 'sessionID':'1016', 'proposalTable': 'Proposal_Field', 'proposalID':'55'}
root.opsimNames = ['ObsHistory']

#filters = ['u','g','r','i','z','y']
filters=['r']

binList=[]

# Fetch the proposal ID values from the database
propids, WFDpropid, DDpropid = runInfo.fetchPropIDs(root.dbAddress['dbAddress'])

# Fetch design and strech specs from DB and scale to the length of the observing run if not 10 years
nvisitBench, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessBench, seeingBench = runInfo.fetchBenchmarks(root.dbAddress['dbAddress'])

# Plotting ranges and normalizations
mag_zpoints = coaddedDepthDesign
seeing_norm = seeingBench
sky_zpoints = skyBrighntessBench
nVisits_plotRange = {'all': 
                     {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}


# Construct a WFD SQL where clause:
wfdWhere = ''
for i,propid in enumerate(WFDpropid):
    if i == 0:
        wfdWhere = wfdWhere+'('+'propID = %s'%propid
    else:
        wfdWhere = wfdWhere+'or propID = %s'%propid
    wfdWhere = wfdWhere+')'


# Metrics per filter and for WFD propID
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'percentileClip':75., 'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0], 'histMax':nVisits_plotRange['all'][f][1]})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'}, plotDict={'normVal':nvisitBench[f], 'percentileClip':80., 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
#    m3 = makeMetricConfig('MedianMetric', params=['5sigma_modified'])
#    m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[f], 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]} )             
#    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1,m2,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\' and "%(f)+wfdWhere, "filter = \'%s\'"%f], kwargs={'simDataFieldRaColName':'', 'simDataFieldDecColName':'', 'simDataFieldIdColName':'Field_fieldID'}, )
    binList.append(binner)


# Number of Visits per observing mode:

for i,f in enumerate(filters):    
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits (full range)'}, plotDict={'units':'Number of Visits', 'histBins':50})
        metricDict = makeDict(m1)
        constraints=[]
        for propid in propids:
            constraints.append("filter = \'%s\' and propID = %s" %(f,propid))
        binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints)
        #binList.append(binner)
                                    
        
# Slew histograms
#m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
#binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
#binList.append(binner)

#m1 = makeMetricConfig('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
#binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
#binList.append(binner)

# Filter Hourglass plots
#m1=makeMetricConfig('HourglassMetric')
#binner = makeBinnerConfig('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750',''])
#binList.append(binner)


# Completeness and Joint Completeness
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)','units':'# visits (WFD only)/ # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.})
# For just WFD proposals
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=makeDict(m1), metadata='WFD', constraints=[""+wfdWhere])
#binList.append(binner)
# For all Observations
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (all) / (# WFD Requested)','units':'# visits (all) / # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.})
binner = makeBinnerConfig('OpsimFieldBinner',metricDict=makeDict(m1),constraints=[""])
#binList.append(binner)



root.binners=makeDict(*binList)


