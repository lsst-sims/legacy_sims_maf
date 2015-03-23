# A MAF configuration file to run SRD-related science performance metrics.

import os
import numpy as np
import healpy as hp
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mConfig(config, runName, dbDir='.', outDir='ScienceOut', nside=128, raCol='fieldRA', decCol='fieldDec',
             benchmark='design', **kwargs):
    """
    A MAF config for SSTAR-like analysis of an opsim run.

    runName must correspond to the name of the opsim output
        (minus '_sqlite.db', although if added this will be stripped off)

    dbDir is the directory the database resides in

    outDir is the output directory for MAF

    benchmark (which can be design, stretch or requested) values used to scale plots of number of visits and coadded depth.
       ('requested' means look up the requested number of visits for the proposal and use that information).
    """

    # Setup Database access
    config.outDir = outDir
    if runName.endswith('_sqlite.db'):
        runName = runName.replace('_sqlite.db', '')
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName
    config.figformat = 'pdf'

    #### Set up parameters for configuring plotting dictionaries and identifying WFD proposals.

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, propTags = opsimdb.fetchPropInfo()

    # Fetch the telescope location from config
    lat, lon, height = opsimdb.fetchLatLonHeight()

    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
    wfdWhere = utils.createSQLWhere('WFD', propTags)
    print '#FYI: WFD "where" clause: %s' %(wfdWhere)
    ddWhere = utils.createSQLWhere('DD', propTags)
    print '#FYI: DD "where" clause: %s' %(ddWhere)

    # Fetch the total number of visits (to create fraction for number of visits per proposal)
    totalNVisits = opsimdb.fetchNVisits()

    # Set up benchmark values, scaled to length of opsim run.
    runLength = opsimdb.fetchRunLength()
    if benchmark == 'requested':
        # Fetch design values for seeing/skybrightness/single visit depth.
        benchmarkVals = utils.scaleBenchmarks(runLength, benchmark='design')
        # Update nvisits with requested visits from config files.
        benchmarkVals['nvisits'] = opsimdb.fetchRequestedNvisits(propId=proptags['WFD'])
        # Calculate expected coadded depth.
        benchmarkVals['coaddedDepth'] = utils.calcCoaddedDepth(benchmarkVals['nvisits'], benchmarkVals['singleVisitDepth'])
    elif (benchmark == 'stretch') or (benchmark == 'design'):
        # Calculate benchmarks for stretch or design.
        benchmarkVals = utils.scaleBenchmarks(runLength, benchmark=benchmark)
        benchmarkVals['coaddedDepth'] = utils.calcCoaddedDepth(benchmarkVals['nvisits'], benchmarkVals['singleVisitDepth'])
    else:
        raise ValueError('Could not recognize benchmark value %s, use design, stretch or requested.' %(benchmark))
    # Check that nvisits is not set to zero (for very short run length).
    for f in benchmarkVals['nvisits']:
        if benchmarkVals['nvisits'][f] == 0:
            print 'Updating benchmark nvisits value in %s to be nonzero' %(f)
            benchmarkVals['nvisits'][f] = 1


    # Set values for min/max range of nvisits for All/WFD and DD plots. These are somewhat arbitrary.
    nvisitsRange = {}
    nvisitsRange['all'] = {'u':[20, 80], 'g':[50,150], 'r':[100, 250],
                           'i':[100, 250], 'z':[100, 300], 'y':[100,300]}
    nvisitsRange['DD'] = {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000],
                          'i':[5000, 8000], 'z':[7000, 10000], 'y':[5000, 8000]}
    # Scale these ranges for the runLength.
    scale = runLength / 10.0
    for prop in nvisitsRange:
        for f in nvisitsRange[prop]:
            for i in [0, 1]:
                nvisitsRange[prop][f][i] = int(np.floor(nvisitsRange[prop][f][i] * scale))

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    ####
    # Add variables to configure the slicer.
    slicerName = 'HealpixSlicer'
    slicerkwargs = {'spatialkey1':raCol, 'spatialkey2':decCol, 'nside':nside}
    if (raCol == 'fieldRA') and (decCol == 'fieldDec'):
        slicermetadata = ''
    else:
        slicermetadata = ' dithered'

    ###
    # Configure some standard summary statistics dictionaries to apply to appropriate metrics.
    # Note there's a complication here, you can't configure multiple versions of a summary metric since that makes a
    # dict with repeated keys.  The workaround is to add blank space (or even more words) to one of
    # the keys, which will be stripped out of the metric class name when the object is instatiated.
    standardStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}, 'CountMetric':{},
                   'NoutliersNsigmaMetric 1':{'metricName':'+3Sigma', 'nSigma':3.},
                   'NoutliersNsigmaMetric 2':{'metricName':'-3Sigma', 'nSigma':-3.}}
    rangeStats={'PercentileMetric 1':{'metricName':'25th%ile', 'percentile':25},
                'PercentileMetric 2':{'metricName':'75th%ile', 'percentile':75},
                'MinMetric':{},
                'MaxMetric':{}}
    allStats = standardStats.copy()
    allStats.update(rangeStats)


    # Set up some 'group' labels
    reqgroup = 'A: Required SRD metrics'
    depthgroup = 'B: Depth per filter'
    uniformitygroup = 'C: Time uniformity'
    seeinggroup = 'D: Seeing distribution'

    histNum = 0
    slicerList = []

    # Calculate the fO metrics for all proposals and WFD only.
    order = 0
    for prop in ('All Prop', 'WFD only'):
        if prop == 'All Prop':
            metadata = 'All proposals' + slicermetadata
            sqlconstraint = ['']
        if prop == 'WFD only':
            metadata = 'WFD only' + slicermetadata
            sqlconstraint = ['%s' %(wfdWhere)]
        # Configure the count metric which is what is used for f0 slicer.
        m1 = configureMetric('CountMetric',
                            kwargs={'col':'expMJD', 'metricName':'fO'},
                            plotDict={'units':'Number of Visits',
                                      'Asky':benchmarkVals['Area'], 'Nvisit':benchmarkVals['nvisitsTotal'],
                                      'xMin':0, 'xMax':1500},
                            summaryStats={'fOArea':{'nside':nside, 'norm':False, 'metricName':'fOArea: Nvisits',
                                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                                          'fOArea':{'nside':nside, 'norm':True, 'metricName':'fOArea: Nvisits fraction',
                                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                                        'fONv':{'nside':nside, 'norm':False, 'metricName':'fONv: Area',
                                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                                        'fONv':{'nside':nside, 'norm':True, 'metricName':'fONv: Area fraction',
                                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']}},
                            displayDict={'group':reqgroup, 'subgroup':'F0', 'displayOrder':order, 'caption':
                                        'FO metric: evaluates the overall efficiency of observing.\n (fOArea = %.1f sq degrees receive at least this many visits out of %d, fONv = this many square degrees out of %.1f receive at least %d visits).'
                                        %(benchmarkVals['Area'], benchmarkVals['nvisitsTotal'],
                                          benchmarkVals['Area'], benchmarkVals['nvisitsTotal'])})
        order += 1
        slicer = configureSlicer('fOSlicer', kwargs=slicerkwargs,
                                 metricDict=makeDict(m1), constraints=sqlconstraint,
                                 metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)


    # Calculate the Rapid Revisit Metric.
    order = 0
    for prop in ('All Prop', 'WFD only'):
        if prop == 'All Prop':
            metadata = 'All proposal' + slicermetadata
            sqlconstraint = ''
        if prop == 'WFD only':
            metadata = 'WFD only' + slicermetadata
            sqlconstraint = wfdWhere
        m1 = configureMetric('RapidRevisitMetric',
                             plotDict={'xMin':0, 'xMax':1},
                             summaryStats = {'FracBelowMetric':{'cutoff':0.5, 'scale':hp.nside2pixarea(nside),
                                                                'metricName':'RAV1'}},
                            displayDict = {'group':reqgroup, 'subgroup':'Rapid Revisit', 'displayOrder':order,
                                           'caption':'Deviation from uniformity for short revisit timescales.'})
        order += 1
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs, constraints = [sqlconstraint],
                                 metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)


    # Trigonometric parallax and proper motion @ r=20 and r=24
    metricList = []
    order = 0
    metricList.append(configureMetric('ParallaxMetric',
                                      kwargs={'metricName':'Parallax 20', 'rmag':20},
                                      displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                                                   'caption':'Parallax precision at r=20. (without refraction).'}))
    order += 1
    metricList.append(configureMetric('ParallaxMetric',
                                      kwargs={'metricName':'Parallax 24', 'rmag':24},
                                    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                                             'caption':'Parallax precision at r=24. (without refraction).'}))
    order += 1
    metricList.append(configureMetric('ParallaxMetric',
                                      kwargs={'metricName':'Parallax Normed', 'rmag':24, 'normalize':True},
                                      plotDict={'xMin':0.4, 'xMax':1},
                                    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                                         'caption':
                                         'Normalized parallax (normalized to optimum observation cadence, 1=optimal, relevant for all magnitudes).'}))
    order += 1
    metricList.append(configureMetric('ProperMotionMetric',
                                      kwargs={'metricName':'ProperMotion 20', 'rmag':20},
                                      plotDict={'percentileClip':95},
                                    displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                                                 'caption':'Proper Motion precision at r=20.'}))
    order += 1
    metricList.append(configureMetric('ProperMotionMetric', kwargs={'rmag':24, 'metricName':'Proper Motion 24'},
                                    plotDict={'percentileClip':95},
                                    displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                                                 'caption':'Proper Motion precision at r=24.'}))
    order += 1
    metricList.append(configureMetric('ProperMotionMetric',
                                      kwargs={'rmag':24,'normalize':True, 'metricName':'Proper Motion Normed'},
                                      plotDict={'xMin':0.25, 'xMax':0.75},
                                displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                                             'caption':'Normalized proper motion at r=24 (normalized to optimum observation cadence - start/end. 1=optimal, relevant to all magnitudes).'}))
    order += 1
    slicer =  configureSlicer(slicerName, kwargs=slicerkwargs,
                            metricDict=makeDict(*metricList), constraints=[''])
    slicerList.append(slicer)

    # Calculate the time uniformity in each filter, for each year.
    order = 0
    yearDates = range(0,int(round(365*runLength))+365,365)
    for i in range(len(yearDates)-1):
        for f in filters:
            metadata = 'Year %d, filter %s' %(i, f) + slicermetadata
            sqlconstraint = ['filter = "%s" and night<=%i' %(f, yearDates[i+1])]
            m1 = configureMetric('UniformityMetric',
                                plotDict={'xMin':0, 'xMax':1},
                                displayDict = {'group':uniformitygroup, 'subgroup':'All Props',
                                               'displayOrder':order+filtorder[f],
                                                'caption':'Deviation from uniformity over entire survey.'})
            slicer = configureSlicer(slicerName, kwargs=slicerkwargs,
                                     metricDict=makeDict(m1), constraints=sqlconstraint,
                                        metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
        order += 20

    # Depth metrics.
    startNum = histNum
    for f in filters:
        propCaption = '%s band, all proposals' %(f)
        sqlconstraint = ['filter = "%s"' %(f)]
        metadata = '%s band' %(f)
        histNum = startNum
        metricList = []
        # Number of visits.
        metricList.append(configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisits'},
                                          plotDict={'units':'Number of visits',
                                                    'xMin':nvisitsRange['all'][f][0],
                                                    'xMax':nvisitsRange['all'][f][1], 'binsize':5},
                                              summaryStats=allStats,
                                              displayDict={'group':depthgroup, 'subgroup':'Nvisits', 'order':filtorder[f],
                                                           'caption':'Number of visits in filter %s, %s.' %(f, propCaption)},
                                              histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                                                         'binsize':5,
                                                         'xMin':nvisitsRange['all'][f][0], 'xMax':nvisitsRange['all'][f][1],
                                                         'legendloc':'upper right'}))
        histNum += 1
        # Coadded depth.
        metricList.append(configureMetric('Coaddm5Metric',
                                            plotDict={'zp':benchmarkVals['coaddedDepth'][f], 'xMin':-0.6, 'xMax':0.6,
                                                        'units':'coadded m5 - %.1f' %benchmarkVals['coaddedDepth'][f]},
                                            summaryStats=allStats,
                                            histMerge={'histNum':histNum, 'legendloc':'upper right',
                                                        'color':colors[f], 'label':'%s' %f, 'binsize':.02},
                                            displayDict={'group':depthgroup, 'subgroup':'Coadded Depth',
                                                        'order':filtorder[f],
                                                        'caption':
                                                        'Coadded depth in filter %s, with %s value subtracted (%.1f), %s. More positive numbers indicate fainter limiting magnitudes.' %(f, benchmark, benchmarkVals['coaddedDepth'][f], propCaption)}))
        histNum += 1
        # Effective time.
        metricList.append(configureMetric('TeffMetric', kwargs={'metricName':'Effective Time',
                                                                'normed':True},
                                          summaryStats=allStats,
                                          histMerge={'histNum':histNum, 'legendLoc':'upper right',
                                                     'color':colors[f], 'label':'%s' %f, 'binsize':0.02},
                                          displayDict={'group':depthgroup, 'subgroup':'Time Eff.', 'order':filtorder[f],
                                                       'caption':'"Time Effective" in filter %s, calculated with fiducial depth %s. Normalized by the fiducial time effective, if every observation was at the fiducial depth.'
                                                       %(f, benchmarkVals['singleVisitDepth'][f])}))
        histNum += 1
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs,
                                 metricDict=makeDict(*metricList), constraints=sqlconstraint,
                                 metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)

    # Good seeing in r/i band metrics, including in first/second years.
    startNum = histNum
    for f in (['r', 'i']):
        for tcolor, tlabel, timespan in zip(['k', 'g', 'r'], ['10 years', '1 year', '2 years'],
                                            ['', ' and night<=365', ' and night<=730']):
        sqlconstraint = ['filter = "%s" %s' %(f, timespan)]
        propCaption = '%s band, all proposals, over %s.' %(f, tlabel)
        metadata = '%s band, %s' %(tlabel)
        histNum = startNum
        metricList = []
        seeing_limit = 0.7
        metricList.append(configureMetric('MinMetric', kwargs={'col':'finSeeing'},
                                          plotDict={'xMin':0.5, 'xMax':0.9},
                                          displayDict={'group':seeinggroup, 'subgroup':'Best Seeing',
                                            'order':filtorder[f],
                                            'caption':'Minimum seeing values in %s.' %(propCaption)},
                                        histMerge={'histNum':histNum, color=tcolor, label='%s %s' %(f, tlabel),
                                            binsize=0.05}))
        histNum += 1
        metricList.append(configureMetric('FracBelowMetric', kwargs={'col':'finSeeing', 'cutoff':seeing_limit},
                                            plotDict={'xMin':0, 'xMax':0.5},
                                            displayDict={'group':seeinggroup, 'subgroup':'Good seeing fraction',
                                                'order':filtorder[f],
                                                'caption':'Fraction of total images with seeing better than %.1f, in %s'
                                                %(seeing_limit, propCaption)},
                                            histMerge={'histNum':histNum, color=tcolor, label='%s %s' %(f, tlabel),
                                                binsize=0.05}))
        histNum += 1
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs,
                                 metricDict=makeDict(*metricList), constraints=sqlconstraint,
                                 metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)

    # Coadded depth (after adding dust extinction)
    # Transient recovery



    config.slicers=makeDict(*slicerList)
    return config
