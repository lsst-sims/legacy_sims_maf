# A MAF config that replicates the SSTAR plots
from lsst.sims.maf.driver.mafConfig import *

# Setup Database access
root.outputDir ='./StarOut_Fields'

small = True # Use the small database included in the repo

if small:
    root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
    root.opsimNames = ['opsim_small']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?
else:
    root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
    root.opsimNames = ['opsim']
    propids = [215, 216, 217, 218, 219]
    WFDpropid = 217
    DDpropid = 219

filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
#filters=['r']

# 10 year Design Specs
nvisitBench={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} 
nVisits_plotRange = {'all': 
                     {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}
mag_zpoints={'u':26.1,'g':27.4, 'r':27.5, 'i':26.8, 'z':26.1, 'y':24.9}
sky_zpoints = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
seeing_norm = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

binList=[]


# Metrics per filter 
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0],
                                    'histMax':nVisits_plotRange['all'][f][1]})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitBench[f], 'ylog':False, 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
    m3 = makeMetricConfig('MedianMetric', params=['5sigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[f], 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]} )             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\'"%f])
    binList.append(binner)

# Metrics per filter, WFD only
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'percentileClip':75., 'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0], 'histMax':nVisits_plotRange['all'][f][1]},
                          histMerge={'histNum':5, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitBench[f], 'percentileClip':80., 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
    m3 = makeMetricConfig('MedianMetric', params=['5sigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[f], 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]},
                          histMerge={'histNum':6, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\' and propID = %d"%(f, WFDpropid)])
    binList.append(binner)


    
# Number of Visits per observing mode:
for i,f in enumerate(filters):    
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
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)','units':'# visits (WFD only)/ # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{},'ExactCompleteMetric':{}})
# For just WFD proposals
binner = makeBinnerConfig('OpsimFieldBinner', metricDict=makeDict(m1), metadata='WFD', constraints=["propID = %d" %(WFDpropid)])
binList.append(binner)
# For all Observations
m1 = makeMetricConfig('CompletenessMetric', plotDict={'xlabel':'# visits (all) / (# WFD Requested)','units':'# visits (all) / # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{},'ExactCompleteMetric':{}})
binner = makeBinnerConfig('OpsimFieldBinner',metricDict=makeDict(m1),constraints=[""])
binList.append(binner)


# The merged histograms for basics 
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['5sigma_modified'],
                          histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'5sigma_modified'},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%f])
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['perry_skybrightness'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'perry_skybrightness'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)
    
    m1 = makeMetricConfig('CountMetric', params=['finSeeing'],
                          histMerge={'histNum':3, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'finSeeing'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['airmass'],
                          histMerge={'histNum':4, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)


root.binners=makeDict(*binList)


