# A MAF config that replicates the SSTAR plots

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mConfig(config, runName, dbDir='.', outputDir='Out', slicerName='HealpixSlicer',
            benchmark='design', **kwargs):
    """
    A MAF config for SSTAR-like analysis of an opsim run.
    
    Uses 'slicerName' for metrics which have the option of using
      [HealpixSlicer, OpsimFieldSlicer, or HealpixSlicerDither] 
      (dithered healpix slicer uses ditheredRA/dec values).

    Uses 'benchmark' (which can be design or stretch) to scale plots of number of visits and coadded depth.
    """

    # Setup Database access
    config.outputDir = outputDir
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName
    config.figformat = 'pdf'

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, WFDpropid, DDpropid = opsimdb.fetchPropIDs()

    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
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


    # Fetch the total number of visits (to create fraction for number of visits per proposal)
    totalNVisits = opsimdb.fetchNVisits()

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    # Set up benchmark values for Stretch and Design, scaled to length of opsim run.
    runLength = opsimdb.fetchRunLength()
    design, stretch = utils.scaleStretchDesign(runLength)
    
    # Set zeropoints and normalization for plots below (and range for nvisits plots).
    if benchmark == 'stretch':
        sky_zpoints = stretch['skybrightness']
        seeing_norm = stretch['seeing']    
        mag_zpoints = stretch['coaddedDepth']
        nvisitBench = stretch['nvisits']
    else:
        sky_zpoints = design['skybrightness']
        seeing_norm = design['seeing']
        mag_zpoints = design['coaddedDepth']
        nvisitBench = design['nvisits']

                
    # Set range of values for visits plots.
    nVisits_plotRange = {'all': 
                         {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200],
                          'z':[100, 250], 'y':[100,250]},
                         'DDpropid': 
                         {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],
                          'z':[7000, 10000], 'y':[5000, 8000]}}

    slicerList=[]

    slicerNames = ['HealpixSlicer', 'HealpixSlicerDither', 'OpsimFieldSlicer']

    if slicerName == 'HealpixSlicer':
        slicerName = 'HealpixSlicer'
        nside = 128
        slicerkwargs = {'nside':nside}
        slicermetadata = ''
    elif slicerName == 'HealpixSlicerDither':
        slicerName = 'HealpixSlicer'
        nside = 128
        slicerkwargs = {'nside':nside, 'spatialkey1':'ditheredRA', 'spatialkey2':'ditheredDec'}
        slicermetadata = 'dithered'
    elif slicerName == 'OpsimFieldSlicer':
        slicerName = 'OpsimFieldSlicer'
        slicerkwargs = {}
        slicermetadata = ''
    else:
        raise ValueError('Do not understand slicerName %s: looking for one of %s' %(slicerName, slicerNames))

    print 'Using %s for generic metrics over the sky.' %(slicerName)

    # Note there's a bug here, you can't configure multiple versions of a summary metric since that makes a
    # dict with repeated keys.  One kinda lame workaround is to add blank space (or even more words) to one of
    # the keys that gets stripped out when the object is instatiated.
    standardStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}, 'CountMetric':{},
                   'NoutliersNsigma 1':{'metricName':'p3Sigma', 'nSigma':3.},
                   'NoutliersNsigma 2':{'metricName':'m3Sigma', 'nSigma':-3.}}

    # Metrics per filter over sky.
    for f in filters:
        m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Nvisits'}, 
                              plotDict={'units':'Number of Visits', 'cbarFormat':'%d',
                                        'xMin':nVisits_plotRange['all'][f][0],
                                        'xMax':nVisits_plotRange['all'][f][1]}, 
                              summaryStats=standardStats,
                              displayDict={'group':'Nvisits', 'subgroup':'All Props', 'order':filtorder[f],
                                          'caption':'Number of visits in filter %s, all proposals.' %(f)})
        
        m2 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisitsRatio'},
                              plotDict={'normVal':nvisitBench[f], 'logScale':False,
                                        'xMin':0.5, 'xMax':1.5,
                                        'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])},
                              displayDict={'group':'Nvisits', 'subgroup':'All Obs, ratio', 'order':filtorder[f],
                                          'caption':
                                          'Number of visits in filter %s divided by %s value (%d), all proposals.'
                                          %(f, benchmark, nvisitBench[f])})
        
        m3 = configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'},
                             summaryStats=standardStats,
                             displayDict={'group':'Single Visit Depth', 'subgroup':'All Props', 'order':filtorder[f],
                                          'caption':'Median single visit depth in filter %s, all proposals.' %(f)})
        
        m4 = configureMetric('Coaddm5Metric', plotDict={'zp':mag_zpoints[f],
                                                        'percentileClip':95.,
                                                        'units':'(coadded m5 - %.1f)' %mag_zpoints[f]},
                             summaryStats=standardStats,
                             histMerge={'histNum':6, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s'%f, 'binsize':.01},
                             displayDict={'group':'CoaddDepth', 'subgroup':'All Props', 'order':filtorder[f],
                                          'caption':
                                          'Coadded depth in filter %s, with %s value subtracted (%.1f), all proposals.'
                                            %(f, benchmark, mag_zpoints[f])})
        
        m5 = configureMetric('MedianMetric', kwargs={'col':'filtSkyBrightness'},
                              plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])},
                              displayDict={'group':'Sky Brightness', 'subgroup':'All Props', 'order':filtorder[f],
                                           'caption':
                                           'Median Sky Brightness in filter %s, with expected zeropoint (%.2f) subtracted, for all proposals.'
                                           %(f, sky_zpoints[f])})        
        m6 = configureMetric('MedianMetric', kwargs={'col':'finSeeing'},
                              plotDict={'normVal':seeing_norm[f],
                                        'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])},
                              displayDict={'group':'Seeing', 'subgroup':'All Props', 'order':filtorder[f],
                                           'caption':
                                           'Median Seeing in filter %s divided by expected value (%.2f), all proposals'
                                           %(f, seeing_norm[f])})                                       
        m7 = configureMetric('MedianMetric', kwargs={'col':'airmass'}, plotDict={'units':'X'},
                             displayDict={'group':'Airmass', 'subgroup':'All Props', 'order':filtorder[f],
                                           'caption':'Median airmass in filter %s, all proposals' %(f)})
        m8 = configureMetric('MaxMetric', kwargs={'col':'airmass'}, plotDict={'units':'X'},
                             displayDict={'group':'Airmass', 'subgroup':'All Props', 'order':filtorder[f],
                                           'caption':'Max airmass in filter %s, all proposals' %(f)})

        metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
        constraints = ['filter = "%s"' %(f)]
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                                  constraints=constraints, metadata=slicermetadata)
        slicerList.append(slicer)

    # Same metrics per filter, but WFD only
    for f in filters:
        m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Nvisits'}, 
                              plotDict={'percentileClip':75., 'units':'Number of Visits',
                                        'cbarFormat':'%d',
                                        'histMin':nVisits_plotRange['all'][f][0],
                                        'histMax':nVisits_plotRange['all'][f][1]},
                              summaryStats=standardStats,
                              displayDict={'group':'Nvisits', 'subgroup':'WFD', 'order':filtorder[f],
                                           'caption':'Number of visits in filter %s, WFD only' %(f)}, 
                              histMerge={'histNum':5, 'legendloc':'upper right',
                                         'color':colors[f],'label':'%s'%f, 'binsize':5})
        m2 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisitsRatio'},
                              plotDict={'normVal':nvisitBench[f],
                                        'xMin':0.5, 'xMax':1.5,
                                        'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])},
                              displayDict={'group':'Nvisits', 'subgroup':'WFD, ratio', 'order':filtorder[f],
                                          'caption':
                                          'Number of visits in filter %s divided by %s value (%d), WFD only.'
                                          %(f, benchmark, nvisitBench[f])})
        m3 = configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'}, summaryStats=standardStats,
                             displayDict={'group':'Single Visit Depth', 'subgroup':'WFD', 'order':filtorder[f],
                                          'caption':'Median single visit depth in filter %s, WFD only.' %(f)})        
        m4 = configureMetric('Coaddm5Metric', plotDict={'zp':mag_zpoints[f], 'percentileClip':95.,
                                                         'units':'(coadded m5 - %.1f)'%mag_zpoints[f]},
                             displayDict={'group':'CoaddDepth', 'subgroup':'WFD', 'order':filtorder[f],
                                          'caption':
                                          'Coadded depth in filter %s, with %s value subtracted (%.1f), WFD only.'
                                            %(f, benchmark, mag_zpoints[f])})        
        m5 = configureMetric('MedianMetric', kwargs={'col':'filtSkyBrightness'},
                              plotDict={'zp':sky_zpoints[f],
                                        'units':'Skybrightness - %.2f' %(sky_zpoints[f])},
                             displayDict={'group':'Sky Brightness', 'subgroup':'WFD', 'order':filtorder[f],
                                           'caption':
                                           'Median Sky Brightness in filter %s, with expected zeropoint (%.2f) subtracted, for WFD only.'
                                           %(f, sky_zpoints[f])})
        m6 = configureMetric('MedianMetric', kwargs={'col':'finSeeing'},
                              plotDict={'normVal':seeing_norm[f],
                                        'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])},
                            displayDict={'group':'Seeing', 'subgroup':'WFD', 'order':filtorder[f],
                                        'caption':
                                        'Median seeing in filter %s, divided by expected value (%.2f), WFD only.'
                                        %(f, seeing_norm[f])}) 
        m7 = configureMetric('MedianMetric', kwargs={'col':'airmass'}, plotDict={'units':'X'},
                             displayDict={'group':'Airmass', 'subgroup':'WFD', 'order':filtorder[f],
                                           'caption':'Median airmass in filter %s, WFD only.' %(f)})
        m8 = configureMetric('MaxMetric', kwargs={'col':'airmass'}, plotDict={'units':'X'},
                             displayDict={'group':'Airmass', 'subgroup':'WFD', 'order':filtorder[f],
                                           'caption':'Max airmass in filter %s, WFD only.' %(f)})
        metricDict = makeDict(m1,m2,m3,m4,m5,m6,m7,m8)
        constraints = ['filter = "%s" and %s' %(f, wfdWhere)]
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                                 constraints=constraints, metadata=slicermetadata + ' (WFD)')
        slicerList.append(slicer)

    
    # Number of Visits per proposal, over sky (repeats WFD). 
    for f in filters:    
        m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisitsPerProp'},
                              plotDict={'units':'Number of Visits', 'cbarFormat':'%d', 'bins':50},
                              displayDict={'group':'NVisits', 'subgroup':'PerProp', 'order':filtorder[f]})
        metricDict = makeDict(m1)
        constraints=[]
        for propid in propids:
            constraints.append("filter = \'%s\' and propID = %s" %(f, propid))
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                                 constraints=constraints, metadata=slicermetadata)
        slicerList.append(slicer)
                                    

    # Slew histograms
    m1 = configureMetric('CountMetric', kwargs={'col':'slewTime'}, plotDict={'logScale':True},
                         displayDict={'group':'Technical'})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'slewTime', 'binsize':5},
                              metricDict=makeDict(m1), constraints=[''], metadata='Slew Time')
    slicerList.append(slicer)

    m1 = configureMetric('CountMetric', kwargs={'col':'slewDist'}, plotDict={'logScale':True},
                         displayDict={'group':'Technical'})
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'slewDist', 'binsize':.05},
                              metricDict=makeDict(m1), constraints=[''], metadata='Slew Distance')
    slicerList.append(slicer)

    # Filter Hourglass plots per year (split to make labelling easier). 
    yearDates = range(0,int(round(365*runLength))+365,365)
    for i in range(len(yearDates)-1):
        constraints = ['night > %i and night <= %i'%(yearDates[i],yearDates[i+1])]
        m1=configureMetric('HourglassMetric', plotDict={'title':'Year %i-%i'%(i,i+1)}
                           displayDict={'group':'Hourglass', 'order':i})
        slicer = configureSlicer('HourglassSlicer', metricDict=makeDict(m1), constraints=constraints)
        slicerList.append(slicer)

    # Completeness and Joint Completeness
    m1 = configureMetric('CompletenessMetric',
                          plotDict={'xlabel':'# visits (WFD only) / (# WFD Requested)',
                                    'units':'# visits (WFD only)/ # WFD',
                                    'xMin':0.5, 'xMax':1.5, 'bins':50},
                          kwargs={'u':nvisitBench['u'], 'g':nvisitBench['g'], 'r':nvisitBench['r'],
                                  'i':nvisitBench['i'], 'z':nvisitBench['z'], 'y':nvisitBench['y']},
                         summaryStats={'TableFractionMetric':{}},
                         displayDict={'group':'Completeness', 'subgroup':'WFD'})
    # For just WFD proposals
    metricDict = makeDict(m1)
    constraints = ['%s' %(wfdWhere)]
    slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict,
                              constraints=constraints, metadata=slicermetadata + ' (WFD)')
    slicerList.append(slicer)
    # For all proposals
    m1 = configureMetric('CompletenessMetric',
                          plotDict={'xlabel':'# visits (all) / (# WFD Requested)',
                                    'units':'# visits (all) / # WFD',
                                    'xMin':0.5, 'xMax':1.5, 'bins':50},
                          kwargs={'u':nvisitBench['u'], 'g':nvisitBench['g'], 'r':nvisitBench['r'],
                                  'i':nvisitBench['i'], 'z':nvisitBench['z'], 'y':nvisitBench['y']},
                          summaryStats={'TableFractionMetric':{}},
                          displayDict = {'group':'Completeness', 'subgroup':'All Props'})
    metricDict = makeDict(m1)
    constraints = ['']
    slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict,
                              constraints=constraints, metadata=slicermetadata)
    slicerList.append(slicer)
    
    # Calculate some basic summary info about run, per filter.
    for f in filters:
        m1 = configureMetric('MeanMetric', kwargs={'col':'finSeeing'})
        m2 = configureMetric('MedianMetric', kwargs={'col':'finSeeing'})
        m3 = configureMetric('MedianMetric', kwargs={'col':'airmass'})
        m4 = configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'})
        metricDict = makeDict(m1, m2, m3, m4)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=['filter = "%s"'%f])
        slicerList.append(slicer)

                              

    # Some other summary statistics over all filters.
    m1 = configureMetric('MeanMetric', kwargs={'col':'slewTime'})
    m2 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'TotalNVisits'})
    m3 = configureMetric('OpenShutterMetric',summaryStats={'IdentityMetric':{}} )
    metricDict = makeDict(m1, m2, m3)
    slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''])
    slicerList.append(slicer)

    # And count number of visits per proposal.
    constraints = ["propID = '%s'"%pid for pid in propids ]
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Number of Visits Per Proposal'},
                         summaryStats={'IdentityMetric':{}, 'NormalizeMetric':{'normVal':totalNVisits}},
                         displayDict={'group':'Technical'})
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1),
                             constraints=constraints)
    slicerList.append(slicer)

    # Count and plot number of visits per night, and calculate average.
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Number of visits per night'}, 
                          summaryStats=standardStats,
                          displayDict={'group':'Technical'})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night','binsize':1},
                             metricDict=makeDict(m1),
                             constraints=[''])
    slicerList.append(slicer)

    

    # fO metrics for all and WFD
    fOnside = 64
    m1 = configureMetric('CountMetric',
                          kwargs={'col':'expMJD', 'metricName':'fO'},
                          plotDict={'units':'Number of Visits', 'xMin':0,
                                    'xMax':1500},
                          summaryStats={'fOArea':{'nside':fOnside},
                                        'fONv':{'nside':fOnside}})
    slicer = configureSlicer('fOSlicer', kwargs={'nside':fOnside},
                              metricDict=makeDict(m1), constraints=['',])
    slicerList.append(slicer)
    slicer = configureSlicer('fOSlicer', kwargs={'nside':fOnside},
                            metricDict=makeDict(m1), constraints=[wfdWhere], metadata='(WFD)')
    slicerList.append(slicer)



                              
    # The merged histograms for basics 
    for f in filters:
        m1 = configureMetric('CountMetric', kwargs={'col':'fiveSigmaDepth'},
                            histMerge={'histNum':1, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f} )
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'fiveSigmaDepth', 'binsize':0.1},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)]) 
        slicerList.append(slicer)

        m1 = configureMetric('CountMetric', kwargs={'col':'filtSkyBrightness'},
                            histMerge={'histNum':2, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f} )
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'filtSkyBrightness', 'binsize':0.1},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        slicerList.append(slicer)
    
        m1 = configureMetric('CountMetric', kwargs={'col':'finSeeing'},
                            histMerge={'histNum':3, 'legendloc':'upper right',
                                       'color':colors[f],'label':'%s'%f} )
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'finSeeing', 'binsize':0.03},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        slicerList.append(slicer)

        m1 = configureMetric('CountMetric', kwargs={'col':'airmass'}, 
                            histMerge={'histNum':4, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f} )
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'airmass', 'binsize':0.01},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)])
        slicerList.append(slicer)


    config.slicers=makeDict(*slicerList)
    return config
