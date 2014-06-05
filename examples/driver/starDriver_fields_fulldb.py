# A MAF config that replicates the SSTAR plots

from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict
from lsst.sims.maf.utils import runInfo

# Setup Database access
root.outputDir ='./StarOut_Fields_full'
root.dbAddress ={'dbAddress':'sqlite:///hewelhog_1016_sqlite.db', 'OutputTable':'Output', 'fieldTable':'Field', 'sessionID':'1016', 'proposalTable': 'Proposal_Field'}
root.opsimName = 'Output'

filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
#filters=['r']

binList=[]

# Fetch the proposal ID values from the database
propids, WFDpropid, DDpropid = runInfo.fetchPropIDs(root.dbAddress['dbAddress'])


# Fetch design and strech specs from DB and scale to the length of the observing run if not 10 years
nvisitDesign, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessDesign, seeingDesign = runInfo.scaleStretchDesign(root.dbAddress['dbAddress'])

# Check how many fields are requested per propID and for all proposals
# Not sure I actually need to use this anywhere yet...
#nFields = runInfo.fetchNFields(root.dbAddress['dbAddress'], propids)


# Plotting ranges and normalizations
mag_zpoints = coaddedDepthDesign
seeing_norm = seeingDesign
sky_zpoints = skyBrighntessDesign
nVisits_plotRange = {'all': 
                     {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}


# Construct a WFD SQL where clause so multiple propIDs can by WFD:
wfdWhere = ''
for i,propid in enumerate(WFDpropid):
    if i == 0:
        wfdWhere = wfdWhere+'('+'propID = %s'%propid
    else:
        wfdWhere = wfdWhere+'or propID = %s'%propid
    wfdWhere = wfdWhere+')'




# Metrics per filter 
for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0],
                                    'histMax':nVisits_plotRange['all'][f][1]})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitDesign[f], 'ylog':False, 'units':'Number of Visits/Design (%d)' %(nvisitDesign[f])})
    m3 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric',kwargs={'m5col':'fivesigma_modified'}, plotDict={'zp':float(mag_zpoints[f]), 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]},
                          histMerge={'histNum':6, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_units':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_units':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\'"%f])
    binList.append(binner)



# Metrics per filter, WFD only
for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'percentileClip':75., 'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0], 'histMax':nVisits_plotRange['all'][f][1]},
                          histMerge={'histNum':5, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitDesign[f], 'percentileClip':80., 'units':'Number of Visits/Design (%d)' %(nvisitDesign[f])})
    m3 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'},plotDict={'zp':float(mag_zpoints[f]), 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]})             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_units':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_units':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\' and "%(f)+wfdWhere])
    binList.append(binner)



# Number of Visits per observing mode:
for f in filters:    
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisitsperprop'}, plotDict={'units':'Number of Visits', 'histBins':50})
        metricDict = makeDict(m1)
        constraints=[]
        for propid in propids:
            constraints.append("filter = \'%s\' and propID = %s" %(f,propid))
        binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=constraints)
        binList.append(binner)
                                    


# Slew histograms
m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
binList.append(binner)

m1 = makeMetricConfig('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
binList.append(binner)

# Filter Hourglass plots
m1=makeMetricConfig('HourglassMetric')
binner = makeBinnerConfig('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750',''])
binList.append(binner)


# Completeness and Joint Completeness
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)','units':'# visits (WFD only)/ # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{}})
# For just WFD proposals
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=makeDict(m1), metadata='WFD', constraints=[wfdWhere])
binList.append(binner)
# For all Observations
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (all) / (# WFD Requested)','units':'# visits (all) / # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{}})
binner = makeBinnerConfig('OpsimFieldBinner',metricDict=makeDict(m1),constraints=[""])
binList.append(binner)


for f in filters:
    m1 = makeMetricConfig('MeanMetric', params=['finSeeing'], summaryStats={'IdentityMetric':{}}) # Use IdentityMetric to pass the results to the summaryStats file.
    binner = makeBinnerConfig('UniBinner', metricDict=makeDict(m1), constraints=['filter = "%s"'%f])
    binList.append(binner)


# The merged histograms for basics 
for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['fivesigma_modified'],
                          histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'fivesigma_modified'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and "%(f)+wfdWhere]) 
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['perry_skybrightness'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'perry_skybrightness'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and "%(f)+wfdWhere])
    binList.append(binner)
    
    m1 = makeMetricConfig('CountMetric', params=['finSeeing'],
                          histMerge={'histNum':3, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'finSeeing'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and "%(f) + wfdWhere])
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['airmass'],
                          histMerge={'histNum':4, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and "%(f) + wfdWhere])
    binList.append(binner)


root.binners=makeDict(*binList)









