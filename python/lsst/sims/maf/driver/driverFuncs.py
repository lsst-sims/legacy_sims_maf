from lsst.sims.maf.driver.mafConfig import MafConfig, makeBinnerConfig, makeMetricConfig, makeDict
import lsst.sims.maf.utils as utils
import os

def install_initialtest(config, runName, filepath, outputDir):
    """Set up the config values."""    

    # Setup Database access (user does not need to edit)
    config.outputDir = outputDir
    sqlitefile = os.path.join(filepath, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName


    ### User edit below here to define binner and metrics desired.
    
    binList=[]
    
    filters = ['r', 'g']
    nside = 64

    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisits'}, 
                            plotDict={'plotMin':0, 'plotMax':200, 'units':'N Visits'},
                            summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
        m2 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'fivesigma_modified'}, 
                            plotDict={'percentileClip':95}, summaryStats={'MeanMetric':{}})
        metricDict = makeDict(m1, m2)
        constraint = 'filter = "%s"' %(f)
        binner = makeBinnerConfig('HealpixBinner',
                                  kwargs={'nside':nside, 'spatialkey1':'fieldRA', 'spatialkey2':'fieldDec'},
                                metricDict=metricDict, constraints=[constraint,])
        config.binners=makeDict(binner)
        binList.append(binner)

    ### End of user edit for simple configs.
    config.binners=makeDict(*binList)
    return config


def install_biggertest(config, runName, filepath, outputDir, binnertype='OpsimFieldBinner'):
    """Set up the config values."""    

    # Setup Database access
    config.outputDir = outputDir
    sqlitefile = os.path.join(filepath, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, WFDpropid, DDpropid = opsimdb.fetchPropIDs()

    # Construct a WFD SQL where clause so multiple propIDs can by WFD:
    wfdWhere = ''
    if len(WFDpropid) == 1:
        wfdWhere = "propID = %d" %(WFDpropid[0])
    else: 
        for i,propid in enumerate(WFDpropid):
            if i == 0:
                wfdWhere = wfdWhere+'('+'propID = %d' %(propid)
            else:
                wfdWhere = wfdWhere+'or propID = %d ' %(propid)
                wfdWhere = wfdWhere+')'


    # Fetch the total number of visits (to create fraction)
    totalNVisits = opsimdb.fetchNVisits()


    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}

    # Set up benchmark values for Stretch and Design, scaled to length of opsim run.
    runLength = opsimdb.fetchRunLength()
    design, stretch = utils.scaleStretchDesign(runLength)
    
    # Set zeropoints and normalization for plots below (and range for nvisits plots).
    sky_zpoints = design['skybrightness']
    seeing_norm = design['seeing']
    
    mag_zpoints = design['coaddedDepth']
    nvisitBench = design['nvisits']
    
    nVisits_plotRange = {'all': 
                         {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200],
                          'z':[100, 250], 'y':[100,250]},
                         'DDpropid': 
                         {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],
                          'z':[7000, 10000], 'y':[5000, 8000]},
                         '216':
                             {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40],
                              'z':[20, 40], 'y':[20, 40]}}

    binList=[]

    if binnertype == 'HealpixBinner':
        nside = 128
        leafsize = 500
        binnerName = 'HealpixBinner'
        binnerkwargs = {'nside':nside}
        binnersetupkwargs = {'leafsize':leafsize}
        binnermetadata = ''
    elif binnertype == 'HealpixBinnerDither':
        nside = 128
        leafsize = 100
        binnerName = 'HealpixBinner'
        binnerkwargs = {'nside':nside, 'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'}
        binnersetupkwargs = {'leafsize':leafsize}
        binnermetadata = 'dithered'
    elif binnertype == 'OpsimFieldBinner':
        binnerName = 'OpsimFieldBinner'
        binnerkwargs = {}
        binnersetupkwargs = {}
        binnermetadata = ''
    else:
        raise ValueError('Do not understand binnertype %s' %(binnertype))
        binner = makeBinnerConfig('HealpixBinner',
                                  kwargs={"nside":nside, 'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'},
                                  metricDict=metricDict, setupKwargs={"leafsize":leafsize},
                                  constraints=["filter = \'%s\'"%f], metadata='dith')
        binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\'"%f])

    # Metrics per filter 
    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                            plotDict={'units':'Number of Visits', 
                                        'histMin':nVisits_plotRange['all'][f][0],
                                        'histMax':nVisits_plotRange['all'][f][1]})
        m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                            plotDict={'normVal':nvisitBench[f], 'ylog':False,
                                      'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
        m3 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'])
        m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[f],
                                                         'percentileClip':95.,
                                                         'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]},
                                histMerge={'histNum':6, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})             
        m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'],
                              plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
        m6 = makeMetricConfig('MedianMetric', params=['finSeeing'],
                              plotDict={'normVal':seeing_norm[f],
                                        'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
        m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_units':'X'})
        m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_units':'X'})
        metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
        constraints = ['filter = "%s"' %(f)]
        binner = makeBinnerConfig(binnerName, kwargs=binnerkwargs, metricDict=metricDict,
                                  setupKwargs=binnersetupkwargs, constraints=constraints, metadata=binnermetadata)
        binList.append(binner)

    # Metrics per filter, WFD only
    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                            plotDict={'percentileClip':75., 'units':'Number of Visits', 
                                        'histMin':nVisits_plotRange['all'][f][0],
                                        'histMax':nVisits_plotRange['all'][f][1]},
                            histMerge={'histNum':5, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})
        m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                            plotDict={'normVal':nvisitBench[f], 'percentileClip':80.,
                                      'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
        m3 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'])
        m4 = makeMetricConfig('Coaddm5Metric', plotDict={'zp':mag_zpoints[f], 'percentileClip':95.,
                                                         'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]})             
        m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'],
                              plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
        m6 = makeMetricConfig('MedianMetric', params=['finSeeing'],
                              plotDict={'normVal':seeing_norm[f],
                                        'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
        m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_units':'X'})
        m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_units':'X'})
        metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
        constraints = ['filter = "%s" and %s' %(f, wfdWhere)]
        binner = makeBinnerConfig(binnerName, kwargs=binnerkwargs, metricDict=metricDict,
                                  setupKwargs=binnersetupkwargs, constraints=constraints, metadata=binnermetadata)
        binList.append(binner)


    
    # Number of Visits per observing mode:
    for f in filters:    
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisitsperprop'},
                              plotDict={'units':'Number of Visits', 'histBins':50})
        metricDict = makeDict(m1)
        constraints=[]
        for propid in propids:
            constraints.append("filter = \'%s\' and propID = %s" %(f, propid))
        binner = makeBinnerConfig(binnerName, kwargs=binnerkwargs, metricDict=metricDict,
                                  setupKwargs=binnersetupkwargs, constraints=constraints, metadata=binnermetadata)
        binList.append(binner)
                                    



    # Slew histograms
    m1 = makeMetricConfig('CountMetric', params=['slewTime'], kwargs={'metadata':'time'})
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewTime'},
                              metricDict=makeDict(m1), constraints=[''] )
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['slewDist'], kwargs={'metadata':'dist'})
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'slewDist'},
                              metricDict=makeDict(m1), constraints=[''] )
    binList.append(binner)

    # Filter Hourglass plots
    m1=makeMetricConfig('HourglassMetric')
    binner = makeBinnerConfig('HourglassBinner', metricDict=makeDict(m1), constraints=['night < 750',''])
    binList.append(binner)


    # Completeness and Joint Completeness
    m1 = makeMetricConfig('CompletenessMetric',
                          plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)',
                                    'units':'# visits (WFD only)/ # WFD',
                                    'plotMin':.5, 'plotMax':1.5, 'histBins':50},
                            kwargs={'u':nvisitBench['u'], 'g':nvisitBench['g'], 'r':nvisitBench['r'],
                                    'i':nvisitBench['i'], 'z':nvisitBench['z'], 'y':nvisitBench['y']},
                            summaryStats={'TableFractionMetric':{}})
    # For just WFD proposals
    metricDict = makeDict(m1)
    constraints = ['%s' %(wfdWhere)]
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict,
                              constraints=constraints, metadata=binnermetadata + ' WFD')
    binList.append(binner)
    # For all Observations
    m1 = makeMetricConfig('CompletenessMetric',
                          plotDict={'xlabel':'# visits (all) / (# WFD Requested)',
                                    'units':'# visits (all) / # WFD',
                                    'plotMin':.5, 'plotMax':1.5, 'histBins':50},
                            kwargs={'u':nvisitBench['u'], 'g':nvisitBench['g'], 'r':nvisitBench['r'],
                                    'i':nvisitBench['i'], 'z':nvisitBench['z'], 'y':nvisitBench['y']},
                            summaryStats={'TableFractionMetric':{}})
    constraints = ['']
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict,
                              constraints=constraints, metadata=binnermetadata)

    
    for f in filters:
        # Calculate the mean seeing for all filters.
        m1 = makeMetricConfig('MeanMetric', params=['finSeeing'], summaryStats={'IdentityMetric':{}})
        m2 = makeMetricConfig('MedianMetric', params=['finSeeing'], summaryStats={'IdentityMetric':{}})
        m3 = makeMetricConfig('MedianMetric', params=['airmass'], summaryStats={'IdentityMetric':{}})
        m4 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'], summaryStats={'IdentityMetric':{}})
        metricDict = makeDict(m1, m2, m3, m4)
        binner = makeBinnerConfig('UniBinner', metricDict=metricDict, constraints=['filter = "%s"'%f])
        binList.append(binner)

    # Some other summary statistics.
    m1 = makeMetricConfig('MeanMetric', params=['slewTime'], summaryStats={'IdentityMetric':{}})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Total Number of Visits'},
                        summaryStats={'IdentityMetric':{}} )
    m3 = makeMetricConfig('OpenShutterMetric',summaryStats={'IdentityMetric':{}} )
    metricDict = makeDict(m1, m2, m3)
    binner = makeBinnerConfig('UniBinner', metricDict=metricDict, constraints=[''])
    binList.append(binner)
    
    # The merged histograms for basics 
    for f in filters:
        m1 = makeMetricConfig('CountMetric', params=['fivesigma_modified'], kwargs={'binsize':10},
                            histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
        binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'fivesigma_modified'},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)]) 
        binList.append(binner)

        m1 = makeMetricConfig('CountMetric', params=['perry_skybrightness'], kwargs={'binsize':0.1},
                            histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
        binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'perry_skybrightness'},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        binList.append(binner)
    
        m1 = makeMetricConfig('CountMetric', params=['finSeeing'], kwargs={'binsize':0.03},
                            histMerge={'histNum':3, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
        binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'finSeeing'},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        binList.append(binner)

        m1 = makeMetricConfig('CountMetric', params=['airmass'], kwargs={'binsize':0.01},
                            histMerge={'histNum':4, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
        binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        binList.append(binner)


    config.binners=makeDict(*binList)
    return config

