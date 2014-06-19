# A MAF config that replicates the SSTAR plots
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Setup Database access
root.outputDir ='./StarOut_Fields'


root.dbAddress ={'dbAddress':'mysql://www:zxcvbnm@localhost/OpsimDB'}
root.opsimNames = ['output_opsimblitz2_1020']
propids = [93,94,95,96,97,98]
WFDpropid = 96 #XXX guessing here...
DDpropid = 94 #XXX Guess




filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'Orange','y':'r'}
#filters=['r']

# 10 year Design Specs
nvisitBench={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} 
nVisits_plotRange = {'all': 
                     #{'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     {'u':[0, 200], 'g':[0,200], 'r':[0, 200], 'i':[0, 200], 'z':[0, 200], 'y':[0,200]},
                     'visits':
                         {'u':[0, 70], 'g':[0,100], 'r':[0, 250], 'i':[0, 250], 'z':[0, 200], 'y':[0,200]  }, 
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}
mag_zpoints={'u':26.1,'g':27.4, 'r':27.5, 'i':26.8, 'z':26.1, 'y':24.9}
sky_zpoints = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
seeing_norm = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

slicerList=[]


# Metrics per filter 
for f in filters:
    m1 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0],
                                    'histMax':nVisits_plotRange['all'][f][1],
                                    'plotMin':nVisits_plotRange['visits'][f][0], 'plotMax':nVisits_plotRange['visits'][f][1]})
    m2 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitBench[f], 'plotMin':0.5, 'plotMax':1.2,
                                    'ylog':False, 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
    m3 = configureMetric('MedianMetric', params=['5sigma_ps'])
    m4 = configureMetric('Coaddm5Metric', kwargs={'m5col':'5sigma_ps'},
                          plotDict={'zp':mag_zpoints[f], 'plotMin':-0.8, 'plotMax':0.8,
                                    'units':'Co-add (m5 - %.1f)'%mag_zpoints[f], 'histMin':23, 'histMax':28},
                          histMerge={'histNum':6, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )             
    m5 = configureMetric('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = configureMetric('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = configureMetric('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = configureMetric('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict, constraints=["filter = \'%s\'"%f])
    slicerList.append(slicer)

# Metrics per filter, WFD only
for f in filters:
    m1 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'percentileClip':75., 'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0], 'histMax':nVisits_plotRange['all'][f][1]},
                          histMerge={'histNum':5, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})
    m2 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitBench[f], 'percentileClip':80., 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
    m3 = configureMetric('MedianMetric', params=['5sigma_ps'])
    m4 = configureMetric('Coaddm5Metric', kwargs={'m5col':'5sigma_ps', 'metricName':'Coaddm5WFD'},
                          plotDict={'zp':mag_zpoints[f], 'plotMin':-0.8, 'plotMax':0.8, 'percentileClip':95.,
                                    'units':'Co-add (m5 - %.1f)'%mag_zpoints[f], 'histMin':23, 'histMax':28},
                           histMerge={'histNum':7, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})             
    m5 = configureMetric('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = configureMetric('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = configureMetric('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = configureMetric('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
    slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict, constraints=["filter = \'%s\' and propID = %d"%(f, WFDpropid)])
    slicerList.append(slicer)


    
# Number of Visits per observing mode:
for f in filters:    
        m1 = configureMetric('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisitsperprop'}, plotDict={'units':'Number of Visits', 'histBins':50}, summaryStats={'MedianMetric':{}, 'MeanMetric':{}, 'RmsMetric':{}, 'CountMetric':{}})
        metricDict = makeDict(m1)
        constraints=[]
        for propid in propids:
            constraints.append("filter = \'%s\' and propID = %s" %(f,propid))
        slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict, constraints=constraints)
        slicerList.append(slicer)
                                    
        
# Slew histograms
m1 = configureMetric('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'slewTime'}, metricDict=makeDict(m1), constraints=[''] )
slicerList.append(slicer)

m1 = configureMetric('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'slewDist'}, metricDict=makeDict(m1), constraints=[''] )
slicerList.append(slicer)

# Filter Hourglass plots
m1=configureMetric('HourglassMetric')
slicer = configureSlicer('HourglassSlicer', metricDict=makeDict(m1), constraints=['night < 750',''])
slicerList.append(slicer)


# Completeness and Joint Completeness
m1 = configureMetric('CompletenessMetric', plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)','units':'# visits (WFD only)/ # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{},'ExactCompleteMetric':{}})
# For just WFD proposals
slicer = configureSlicer('OpsimFieldSlicer', metricDict=makeDict(m1), metadata='WFD', constraints=["propID = %d" %(WFDpropid)])
slicerList.append(slicer)
# For all Observations
m1 = configureMetric('CompletenessMetric', plotDict={'xlabel':'# visits (all) / (# WFD Requested)','units':'# visits (all) / # WFD','plotMin':.5, 'plotMax':1.5, 'histBins':50}, kwargs={'u':56., 'g':80., 'r':184., 'i':184.,"z":160.,"y":160.}, summaryStats={'TableFractionMetric':{},'ExactCompleteMetric':{}})
slicer = configureSlicer('OpsimFieldSlicer',metricDict=makeDict(m1),constraints=[""])
slicerList.append(slicer)


# The merged histograms for basics 
for f in filters:
    m1 = configureMetric('CountMetric', params=['5sigma_ps'], plotDict={'histMin':20, 'histMax':26},
                          histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'5sigma_ps'},
                              metricDict=makeDict(m1), constraints=["filter = '%s'and propID = %i"%(f,WFDpropid)]) 
    slicerList.append(slicer)

    m1 = configureMetric('CountMetric', params=['perry_skybrightness'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'perry_skybrightness'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and propID = %i"%(f,WFDpropid)])
    slicerList.append(slicer)
    
    m1 = configureMetric('CountMetric', params=['finSeeing'],
                          histMerge={'histNum':3, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'finSeeing'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and propID = %i"%(f,WFDpropid)])
    slicerList.append(slicer)

    m1 = configureMetric('CountMetric', params=['airmass'],
                          histMerge={'histNum':4, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'airmass'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' and propID = %i"%(f,WFDpropid)])
    slicerList.append(slicer)


root.slicers=makeDict(*slicerList)


