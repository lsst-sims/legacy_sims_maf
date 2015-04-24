# A MAF configuration file to run scheduler validation metrics.

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils
import numpy as np


def mConfig(config, runName, dbDir='.', outDir='Out', benchmark='design', **kwargs):
    """
    A MAF config for SSTAR-like analysis of an opsim run.

    runName must correspond to the name of the opsim output
        (minus '_sqlite.db', although if added this will be stripped off)

    dbDir is the directory the database resides in

    outDir is the output directory for MAF

    benchmark (which can be design, stretch or requested) values used to scale plots of number of visits and coadded depth.
       ('requested' means look up the requested number of visits for the proposal and use that information).
    """
    #config.mafComment = 'Scheduler Validation'

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
    DDpropid = propTags['DD']
    WFDpropid = propTags['WFD']

    # Fetch the telescope location from config
    lat, lon, height = opsimdb.fetchLatLonHeight()

    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
    wfdWhere = utils.createSQLWhere('WFD', propTags)
    print '#FYI: WFD "where" clause: %s' %(wfdWhere)
    ddWhere = utils.createSQLWhere('DD', propTags)
    print '#FYI: DD "where" clause: %s' %(ddWhere)

    # Set up benchmark values, scaled to length of opsim run. These are applied to 'all' and 'WFD' plots.
    runLength = opsimdb.fetchRunLength()
    if benchmark == 'requested':
        # Fetch design values for seeing/skybrightness/single visit depth.
        benchmarkVals = utils.scaleBenchmarks(runLength, benchmark='design')
        # Update nvisits with requested visits from config files.
        benchmarkVals['nvisits'] = opsimdb.fetchRequestedNvisits(propId=WFDpropid)
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

    # Generate approximate benchmark values for DD.
    if len(DDpropid) > 0:
        benchmarkDDVals = {}
        benchmarkDDVals = utils.scaleBenchmarks(runLength, benchmark='design')
        benchmarkDDVals['nvisits'] = opsimdb.fetchRequestedNvisits(propId=DDpropid)
        #benchmarkDDVals['coaddedDepth'] = utils.calcCoaddedDepth(benchmarkDDVals['nvisits'], benchmarkDDVals['singleVisitDepth'])
        benchmarkDDVals['coaddedDepth'] = {'u':28.5, 'g':28.5, 'r':28.5, 'i':28.5, 'z':28.0, 'y':27.0}

    # Set values for min/max range of nvisits for All/WFD and DD plots. These are somewhat arbitrary.
    nvisitsRange = {}
    nvisitsRange['all'] = {'u':[20, 80], 'g':[50,150], 'r':[100, 250],
                           'i':[100, 250], 'z':[100, 300], 'y':[100,300]}
    nvisitsRange['DD'] = {'u':[3000, 7000], 'g':[1000, 7000], 'r':[1000, 7000],
                          'i':[1000, 7000], 'z':[1000, 7000], 'y':[1000, 7000]}
    #for f in benchmarkDDVals['nvisits']:
    #    nvisitsRange['DD'][f][0] = np.min([benchmarkDDVals['nvisits'] - 2000, 0])
    #    nvisitsRange['DD'][f][1] = benchmarkDDVals['nvisits'] + 2000

    # Scale these nvisit ranges for the runLength.
    scale = runLength / 10.0
    for prop in nvisitsRange:
        for f in nvisitsRange[prop]:
            for i in [0, 1]:
                nvisitsRange[prop][f][i] = int(np.floor(nvisitsRange[prop][f][i] * scale))

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors = {'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    ####
    # Add variables to configure the slicer (in case we want to change it in the future).
    slicerName = 'OpsimFieldSlicer'
    slicerkwargs = {}
    slicermetadata = ''
    # For a few slicer/metric combos, we want to only create histograms (not skymaps or power spectra), but keep
    #  the rest of slicerkwargs.
    onlyHist = {'plotFuncs':'plotHistogram'}
    onlyHist.update(slicerkwargs)

    ###
    # Configure some standard summary statistics dictionaries to apply to appropriate metrics.
    # Note there's a complication here, you can't configure multiple versions of a summary metric since that makes a
    # dict with repeated keys.  The workaround is to add blank space (or even more words) to one of
    # the keys, which will be stripped out of the metric class name when the object is instatiated.
    standardStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}, 'CountMetric':{},
                   'NoutliersNsigmaMetric 1':{'metricName':'N(+3Sigma)', 'nSigma':3.},
                   'NoutliersNsigmaMetric 2':{'metricName':'N(-3Sigma)', 'nSigma':-3.}}
    rangeStats={'PercentileMetric 1':{'metricName':'25th%ile', 'percentile':25},
                'PercentileMetric 2':{'metricName':'75th%ile', 'percentile':75},
                'MinMetric':{},
                'MaxMetric':{}}
    allStats = standardStats.copy()
    allStats.update(rangeStats)

    # Standardize a couple of labels (for ordering purposes in showMaf).
    summarygroup = 'A: Summary'
    completenessgroup = 'B: Completeness'
    nvisitgroup = 'C: NVisits'
    nvisitOpsimgroup = 'D: NVisits (per prop)'
    coaddeddepthgroup = 'E: Coadded depth'
    airmassgroup = 'F: Airmass'
    seeinggroup = 'G: Seeing'
    skybrightgroup = 'H: SkyBrightness'
    singlevisitdepthgroup = 'I: Single Visit Depth'
    houranglegroup = 'J: Hour Angle'
    rotatorgroup = 'K: Rotation Angles'
    dist2moongroup = 'L: Distance to Moon'
    hourglassgroup = 'M: Hourglass'
    filtergroup = 'N: Filter Changes'
    slewgroup = 'O: Slew'

    # Fetch the total number of visits (to create fraction for number of visits per proposal)
    totalNVisits = opsimdb.fetchNVisits()
    totalSlewN = opsimdb.fetchTotalSlewN()

    ####
    # Start specifying metrics and slicers for MAF to run.

    slicerList=[]
    histNum = 0

    ## Metrics calculating values across the sky (opsim slicer).
    # Loop over a set of standard analysis metrics, for All Proposals, WFD only, and DD only.

    startNum = histNum
    for i, prop in enumerate(['All Props', 'WFD', 'DD']):
        startNum += 100
        for f in filters:
            # Set some per-proposal information.
            if prop == 'All Props':
                subgroup = 'All Props'
                propCaption = ' for all proposals'
                metadata = '%s band, all props' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s"' %(f)]
                nvisitsMin = nvisitsRange['all'][f][0]
                nvisitsMax = nvisitsRange['all'][f][1]
                mag_zp = benchmarkVals['coaddedDepth'][f]
            elif prop == 'WFD':
                subgroup = 'WFD'
                propCaption = ' for all WFD proposals'
                metadata = '%s band, WFD' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s" and %s' %(f, wfdWhere)]
                nvisitsMin = nvisitsRange['all'][f][0]
                nvisitsMax = nvisitsRange['all'][f][1]
                mag_zp = benchmarkVals['coaddedDepth'][f]
            elif prop == 'DD':
                if len(DDpropid) == 0:
                    continue
                subgroup = 'DD'
                propCaption = ' for all DD proposals'
                metadata = '%s band, DD' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s" and %s' %(f, ddWhere)]
                nvisitsMin = nvisitsRange['DD'][f][0]
                nvisitsMax = nvisitsRange['DD'][f][1]
                mag_zp = benchmarkDDVals['coaddedDepth'][f]
            # Reset histNum (for merged histograms, merged over all filters).
            histNum = startNum
            # Configure the metrics to run for this sql constraint (all proposals/wfd and filter combo).
            metricList = []
            # Count the total number of visits.
            metricList.append(configureMetric('CountMetric',
                                              kwargs={'col':'expMJD', 'metricName':'Nvisits'},
                                              plotDict={'units':'Number of Visits',
                                                'xMin':nvisitsMin,
                                                'xMax':nvisitsMax, 'binsize':5},
                                              summaryStats=allStats,
                                              displayDict={'group':nvisitgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                           'caption':'Number of visits in filter %s, %s.' %(f, propCaption)},
                                              histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                                                         'binsize':5, 'xMin':nvisitsMin, 'xMax':nvisitsMax,
                                                         'legendloc':'upper right'}))
            histNum += 1
            # Calculate the coadded five sigma limiting magnitude (normalized to a benchmark).
            metricList.append(configureMetric('Coaddm5Metric',
                                              plotDict={'zp':mag_zp, 'xMin':-0.6, 'xMax':0.6,
                                                        'units':'coadded m5 - %.1f' %mag_zp},
                                            summaryStats=allStats,
                                            histMerge={'histNum':histNum, 'legendloc':'upper right',
                                                        'color':colors[f], 'label':'%s' %f, 'binsize':.02},
                                            displayDict={'group':coaddeddepthgroup, 'subgroup':subgroup,
                                                        'order':filtorder[f],
                                                        'caption':
                                                        'Coadded depth in filter %s, with %s value subtracted (%.1f), %s. More positive numbers indicate fainter limiting magnitudes.' %(f, benchmark, mag_zp, propCaption)}))
            histNum += 1
            # Only calculate the rest of these metrics for NON-DD proposals.
            if prop != 'DD':
                # Count the number of visits as a ratio against a benchmark value, for 'all' and 'WFD'.
                metricList.append(configureMetric('CountRatioMetric',
                                              kwargs={'col':'expMJD', 'normVal':benchmarkVals['nvisits'][f],
                                                      'metricName':'NVisitsRatio'},
                                              plotDict={ 'binsize':0.05,'cbarFormat':'%2.2f',
                                                    'colorMin':0.5, 'colorMax':1.5, 'xMin':0.475, 'xMax':1.525,
                                                    'units':'Number of Visits/Benchmark (%d)' %(benchmarkVals['nvisits'][f])},
                                              displayDict={'group':nvisitgroup, 'subgroup':'%s, ratio' %(subgroup),
                                                           'order':filtorder[f],
                                                           'caption': 'Number of visits in filter %s divided by %s value (%d), %s.'
                                                     %(f, benchmark, benchmarkVals['nvisits'][f], propCaption)},
                                              histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                                                         'xlabel':'Number of visits / benchmark',
                                                         'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                                                         'legendloc':'upper right'}))
                histNum += 1
                # Calculate the median individual visit five sigma limiting magnitude (individual image depth).
                metricList.append(configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'},
                                    summaryStats=standardStats,
                                    displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':'Median single visit depth in filter %s, %s.' %(f, propCaption)}))
                # Calculate the median individual visit sky brightness (normalized to a benchmark).
                metricList.append(configureMetric('MedianMetric',
                                                kwargs={'col':'filtSkyBrightness'},
                                                plotDict={'zp':benchmarkVals['skybrightness'][f],
                                                          'units':'Skybrightness - %.2f' %(benchmarkVals['skybrightness'][f]),
                                                          'xMin':-2, 'xMax':1},
                                                displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':
                                                'Median Sky Brightness in filter %s with expected zeropoint (%.2f) subtracted, %s. Fainter sky brightness values are more positive numbers.'
                                                %(f, benchmarkVals['skybrightness'][f], propCaption)}))
                # Calculate the median delivered seeing.
                metricList.append(configureMetric('MedianMetric', kwargs={'col':'finSeeing'},
                                        plotDict={'normVal':benchmarkVals['seeing'][f],
                                                    'units':'Median Seeing/(Expected seeing %.2f)'%(benchmarkVals['seeing'][f])},
                                        displayDict={'group':seeinggroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                    'caption':
                                                    'Median Seeing in filter %s divided by expected value (%.2f), %s.'
                                                    %(f, benchmarkVals['seeing'][f], propCaption)}))
                # Calculate the median airmass.
                metricList.append(configureMetric('MedianMetric',
                                                kwargs={'col':'airmass'},
                                                plotDict={'units':'X'},
                                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                            'caption':'Median airmass in filter %s, %s.' %(f, propCaption)}))
                # Calculate the median normalized airmass.
                metricList.append(configureMetric('MedianMetric',
                                                kwargs={'col':'normairmass'},
                                                plotDict={'units':'X'},
                                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                            'caption':'Median normalized airmass (airmass divided by the minimum airmass a field could reach) in filter %s, %s.'
                                                            %(f, propCaption)}))
                # Calculate the maximum airmass.
                metricList.append(configureMetric('MaxMetric',
                                                kwargs={'col':'airmass'},
                                                plotDict={'units':'X'},
                                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':'Max airmass in filter %s, %s.' %(f, propCaption)}))
                # Calculate the mean of the hour angle.
                metricList.append(configureMetric('MeanMetric', kwargs={'col':'HA'},
                                                  plotDict={'xMin':-6, 'xMax':6},
                                                  displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                               'caption':'Full Range of the Hour Angle in filter %s, %s.'
                                                               %(f, propCaption)}))
                # Calculate the Full Range of the hour angle.
                metricList.append(configureMetric('FullRangeMetric', kwargs={'col':'HA'},
                                                  plotDict={'xMin':0, 'xMax':12},
                                                  displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                               'caption':'Full Range of the Hour Angle in filter %s, %s.'
                                                               %(f, propCaption)}))
                # Calculate the RMS of the position angle
                metricList.append(configureMetric('RmsAngleMetric', kwargs={'col':'rotSkyPos'},
                                                  plotDict={'xMin':0, 'xMax':float(np.pi)},
                                                  displayDict={'group':rotatorgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                               'caption':'RMS of the position angle (angle between "up" in the camera and north on the sky) in filter %s, %s.'
                                                               %(f, propCaption)}))
            metricDict = makeDict(*metricList)
            slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                                     constraints=sqlconstraint, metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Tack on an extra copy of Nvisits with a cumulative histogram for WFD.
            if prop == 'WFD':
                metric = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Nvisits cumulative'},
                                              plotDict={'units':'Number of Visits',
                                                        'xMin':0,
                                                        'xMax':nvisitsMax, 'binsize':5,
                                                        'cumulative':-1},
                                              displayDict={'group':nvisitgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                           'caption':'Cumulative number of visits in filter %s, %s.'
                                                            %(f, propCaption)},
                                              histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                                                         'binsize':5, 'xMin':0, 'xMax':nvisitsMax, 'legendloc':'upper right',
                                                         'cumulative':-1})
                histNum += 1
                slicer = configureSlicer(slicerName, kwargs=onlyHist, metricDict=makeDict(*[metric]),
                                        constraints=sqlconstraint, metadata=metadata, metadataVerbatim=True)
                slicerList.append(slicer)

    # Count the number of visits in all filters together, WFD only.
    metricList =[]
    # Make the reverse cumulative histogram
    metricList.append(configureMetric('CountMetric',
                                      kwargs={'col':'expMJD', 'metricName':'Nvisits, all filters, cumulative'},
                                      plotDict={'units':'Number of Visits', 'binsize':5, 'cumulative':-1,
                                                'xMin':500, 'xMax':1500},
                                      displayDict={'group':nvisitgroup, 'subgroup':'WFD', 'order':0,
                                                   'caption':'Number of visits all filters, WFD only'}))
    # Regular Histogram
    metricList.append(configureMetric('CountMetric',
                                      kwargs={'col':'expMJD', 'metricName':'Nvisits, all filters'},
                                      plotDict={'units':'Number of Visits', 'binsize':5, 'cumulative':False,
                                                'xMin':500, 'xMax':1500},
                                      summaryStats=allStats,
                                      displayDict={'group':nvisitgroup, 'subgroup':'WFD', 'order':0,
                                                   'caption':'Number of visits all filters, WFD only'}))
    slicer = configureSlicer(slicerName, kwargs=onlyHist, metricDict=makeDict(*metricList),
                                     constraints=[wfdWhere],
                                     metadata='All filters WFD: histogram only', metadataVerbatim=True)
    slicerList.append(slicer)

    # Count the number of visits per filter for each individual proposal, over the sky.
    #  The min/max limits for these plots are allowed to float, so that we can really see what's going on in each proposal.
    propOrder = 0
    for propid in propids:
        for f in filters:
            # Count the number of visits.
            m1 = configureMetric('CountMetric',
                                kwargs={'col':'expMJD', 'metricName':'NVisits Per Proposal'},
                                summaryStats=standardStats,
                                plotDict={'units':'Number of Visits', 'plotMask':True,
                                          'binsize':5},
                                displayDict={'group':nvisitOpsimgroup, 'subgroup':'%s'%(propids[propid]),
                                             'order':filtorder[f] + propOrder,
                                             'caption':'Number of visits per opsim field in %s filter, for %s.'
                                             %(f, propids[propid])},
                                histMerge={'histNum':histNum, 'legendloc':'upper right', 'color':colors[f],
                                           'label':'%s' %f, 'binsize':5})
            metricDict = makeDict(m1)
            sqlconstraint = ['filter = "%s" and propID = %s' %(f,propid)]
            slicer = configureSlicer(slicerName, kwargs=slicerkwargs,
                                     metricDict=metricDict,
                                     constraints=sqlconstraint,
                                     metadata='%s band, %s' %(f, propids[propid]),
                                     metadataVerbatim=True)
            slicerList.append(slicer)
        propOrder += 100
        histNum += 1

    # Run for combined WFD proposals if there's more than one. Similar to above, but with different nvisits limits.
    if len(WFDpropid) > 1:
        for f in filters:
            m1 = configureMetric('CountMetric',
                                 kwargs={'col':'expMJD', 'metricName':'NVisits Per Proposal'},
                                 summaryStats=standardStats,
                                 plotDict={'units':'Number of Visits', 'binsize':5},
                                 displayDict={'group':nvisitOpsimgroup, 'subgroup':'WFD',
                                              'order':filtorder[f] + propOrder,
                                              'caption':'Number of visits per opsim field in %s filter, for WFD.' %(f)},
                                 histMerge={'histNum':histNum, 'legendloc':'upper right',
                                            'color':colors[f], 'label':'%s' %f, 'binsize':5})
            metricDict = makeDict(m1)
            sqlconstraint = ['filter = "%s" and %s' %(f, wfdWhere)]
            slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict, constraints=sqlconstraint,
                                     metadata='%s band, WFD' %(f), metadataVerbatim=True)
            slicerList.append(slicer)
        histNum += 1

    # Calculate the Completeness and Joint Completeness for all proposals and WFD only.
    for prop in ('All Props', 'WFD'):
        if prop == 'All Props':
            subgroup = 'All Props'
            metadata = 'All proposals'
            sqlconstraint = ['']
            xlabel = '# visits (All Props) / (# WFD %s value)' %(benchmark)
        if prop == 'WFD':
            subgroup = 'WFD'
            metadata = 'WFD only'
            sqlconstraint = ['%s' %(wfdWhere)]
            xlabel = '# visits (WFD) / (# WFD %s value)' %(benchmark)
        # Configure completeness metric.
        m1 = configureMetric('CompletenessMetric',
                            plotDict={'xlabel':xlabel,
                                        'units':xlabel,
                                        'xMin':0.5, 'xMax':1.5, 'bins':50},
                            kwargs={'u':benchmarkVals['nvisits']['u'], 'g':benchmarkVals['nvisits']['g'],
                                    'r':benchmarkVals['nvisits']['r'], 'i':benchmarkVals['nvisits']['i'],
                                    'z':benchmarkVals['nvisits']['z'], 'y':benchmarkVals['nvisits']['y']},
                            summaryStats={'TableFractionMetric':{}},
                            displayDict={'group':completenessgroup, 'subgroup':subgroup})
        metricDict = makeDict(m1)
        slicer = configureSlicer(slicerName, kwargs=slicerkwargs, metricDict=metricDict,
                                constraints=sqlconstraint, metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)

    ## End of all-sky metrics.

    ## Hourglass metric.
    # Calculate Filter Hourglass plots per year (split to make labelling easier).
    yearDates = range(0,int(round(365*runLength))+365,365)
    for i in range(len(yearDates)-1):
        constraints = ['night > %i and night <= %i'%(yearDates[i],yearDates[i+1])]
        m1=configureMetric('HourglassMetric', kwargs={'lat':lat*np.pi/180.,
                                                      'lon':lon*np.pi/180. , 'elev':height},
                           displayDict={'group':hourglassgroup, 'subgroup':'Yearly', 'order':i})
        slicer = configureSlicer('HourglassSlicer', metricDict=makeDict(m1), constraints=constraints,
                                 metadata='Year %i-%i' %(i, i+1), metadataVerbatim=True)
        slicerList.append(slicer)

    ## Histograms of individual output values of Opsim. (one-d slicers).

    # Histograms per filter for All & WFD only (generally used to produce merged histograms).
    startNum = histNum
    for i, prop in enumerate(['All Props', 'WFD']):
        for f in filters:
            # Set some per-proposal information.
            if prop == 'All Props':
                subgroup = 'All Props'
                propCaption = ' for all proposals.'
                metadata = '%s band, all props' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s"' %(f)]
                # Reset histNum to starting value (to combine filters).
                histNum = startNum
            elif prop == 'WFD':
                subgroup = 'WFD'
                propCaption = ' for all WFD proposals.'
                metadata = '%s band, WFD' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s" and %s' %(f, wfdWhere)]
                # Reset histNum to starting value (to combine filters).
                histNum = startNum + 20
            # Set up metrics and slicers for histograms.
            # Histogram the individual visit five sigma limiting magnitude (individual image depth).
            m1 = configureMetric('CountMetric',
                                kwargs={'col':'fiveSigmaDepth', 'metricName':'Single Visit Depth Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s'%f},
                                displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the single visit depth in %s band, %s.' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'fiveSigmaDepth', 'binsize':0.05},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit sky brightness.
            m1 = configureMetric('CountMetric', kwargs={'col':'filtSkyBrightness', 'metricName':'Sky Brightness Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s'%f},
                                displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the sky brightness in %s band, %s.' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'filtSkyBrightness', 'binsize':0.1,
                                                        'binMin':16, 'binMax':23},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit seeing.
            m1 = configureMetric('CountMetric', kwargs={'col':'finSeeing', 'metricName':'Seeing Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s'%f},
                                displayDict={'group':seeinggroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the seeing in %s band, %s.' %(f, propCaption)} )
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'finSeeing', 'binsize':0.02},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit airmass values.
            m1 = configureMetric('CountMetric', kwargs={'col':'airmass', 'metricName':'Airmass Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s' %f, 'xMin':1.0, 'xMax':2.0},
                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the airmass in %s band, %s' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'airmass', 'binsize':0.01},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit normalized airmass values.
            m1 = configureMetric('CountMetric', kwargs={'col':'normairmass', 'metricName':'Normalized Airmass Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s' %f, 'xMin':1.0, 'xMax':2.0},
                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the normalized airmass in %s band, %s' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'normairmass', 'binsize':0.01},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit hour angle values.
            m1 = configureMetric('CountMetric', kwargs={'col':'HA', 'metricName':'Hour Angle Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s' %f, 'xMin':-6., 'xMax':6},
                                displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the hour angle in %s band, %s' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'HA', 'binsize':0.1},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the sky position angles (rotSkyPos)
            m1 = configureMetric('CountMetric', kwargs={'col':'rotSkyPos', 'metricName':'Position Angle Histogram'},
                                 histMerge={'histNum':histNum, 'legendloc':'upper right',
                                            'color':colors[f], 'label':'%s' %f, 'xMin':0., 'xMax':float(np.pi*2.)},
                                displayDict={'group':rotatorgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                             'caption':'Histogram of the position angle (in radians) in %s band, %s. The position angle is the angle between "up" in the image and North on the sky.' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'rotSkyPos', 'binsize':0.05},
                                     metricDict=makeDict(m1), constraints=sqlconstraint,
                                     metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)
            # Histogram the individual visit distance to moon values.
            m1 = configureMetric('CountMetric', kwargs={'col':'dist2Moon', 'metricName':'Distance to Moon Histogram'},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                        'color':colors[f], 'label':'%s'%f,
                                        'xMin':float(np.radians(15.)), 'xMax':float(np.radians(180.))},
                                displayDict={'group':dist2moongroup, 'subgroup':subgroup, 'order':filtorder[f],
                                            'caption':'Histogram of the distance between the field and the moon (in radians) in %s band, %s' %(f, propCaption)})
            histNum += 1
            slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'dist2Moon', 'binsize':0.05},
                                    metricDict=makeDict(m1), constraints=sqlconstraint,
                                    metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)

   # Slew histograms (time and distance).
    m1 = configureMetric('CountMetric', kwargs={'col':'slewTime', 'metricName':'Slew Time Histogram'},
                         plotDict={'logScale':True, 'ylabel':'Count'},
                         displayDict={'group':slewgroup, 'subgroup':'Slew Histograms',
                                      'caption':'Histogram of slew times for all visits.'})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'slewTime', 'binsize':5},
                              metricDict=makeDict(m1), constraints=[''])
    slicerList.append(slicer)
    m1 = configureMetric('CountMetric', kwargs={'col':'slewDist', 'metricName':'Slew Distance Histogram'},
                         plotDict={'logScale':True, 'ylabel':'Count'},
                         displayDict={'group':slewgroup, 'subgroup':'Slew Histograms',
                                      'caption':'Histogram of slew distances for all visits.'})
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceColName":'slewDist', 'binsize':.05},
                              metricDict=makeDict(m1), constraints=[''])
    slicerList.append(slicer)


    # Plots per night -- the number of visits and the open shutter time fraction.
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisits'},
                          summaryStats=allStats,
                          displayDict={'group':summarygroup, 'subgroup':'3: Obs Per Night',
                                       'caption':'Number of visits per night.'})
    m2 = configureMetric('OpenShutterFractionMetric',
                         summaryStats=allStats,
                         displayDict={'group':summarygroup, 'subgroup':'3: Obs Per Night',
                                      'caption':'Open shutter fraction per night. This compares the on-sky image time against the on-sky time + slews/filter changes/readout, but does not include downtime due to weather.'})
    m3 = configureMetric('NChangesMetric', kwargs={'col':'filter', 'metricName':'Filter Changes'},
                         summaryStats=allStats,
                         plotDict={'units':'Number of Filter Changes'},
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                     'caption':'Number of filter changes per night.'})
    m4 = configureMetric('MinTimeBetweenStatesMetric',
                         kwargs={'changeCol':'filter'},
                         plotDict={'yMin':0, 'yMax':120},
                         summaryStats=allStats,
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                      'caption':'Minimum time between filter changes, in minutes.'})
    m5 = configureMetric('NStateChangesFasterThanMetric',
                         kwargs={'changeCol':'filter', 'cutoff':10},
                         summaryStats=allStats,
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                      'caption':'Number of filter changes, where the time between filter changes is shorter than 10 minutes, per night.'})
    m6 = configureMetric('NStateChangesFasterThanMetric',
                         kwargs={'changeCol':'filter', 'cutoff':20},
                         summaryStats=allStats,
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                      'caption':'Number of filter changes, where the time between filter changes is shorter than 20 minutes, per night.'})
    m7 = configureMetric('MaxStateChangesWithinMetric',
                         kwargs={'changeCol':'filter', 'timespan':10},
                         summaryStats=allStats,
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                      'caption':'Max number of filter changes within a window of 10 minutes, per night.'})
    m8 = configureMetric('MaxStateChangesWithinMetric',
                         kwargs={'changeCol':'filter', 'timespan':20},
                         summaryStats=allStats,
                         displayDict={'group':filtergroup, 'subgroup':'Per Night',
                                      'caption':'Max number of filter changes within a window of 20 minutes, per night.'})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night','binsize':1},
                             metricDict=makeDict(m1, m2, m3, m4, m5, m6, m7, m8),
                             constraints=[''], metadata='Per night', metadataVerbatim=True)
    slicerList.append(slicer)

    ## Unislicer (single number) metrics.
    order = 0
    m1 = configureMetric('NChangesMetric', kwargs={'col':'filter', 'metricName':'Total Filter Changes'},
                         displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                                      'caption':'Total filter changes over survey'})
    order += 1
    m2 = configureMetric('MinTimeBetweenStatesMetric', kwargs={'changeCol':'filter'},
                        displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                                     'caption':'Minimum time between filter changes, in minutes.'})
    order += 1 
    m3 = configureMetric('NStateChangesFasterThanMetric', kwargs={'changeCol':'filter', 'cutoff':10},
                        displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                        'caption':'Number of filter changes faster than 10 minutes over the entire survey.'})
    order += 1
    m4 = configureMetric('NStateChangesFasterThanMetric', kwargs={'changeCol':'filter', 'cutoff':20},
                        displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                        'caption':'Number of filter changes faster than 20 minutes over the entire survey.'})
    order += 1
    m5 = configureMetric('MaxStateChangesWithinMetric',
                         kwargs={'changeCol':'filter', 'timespan':10},
                         displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                                      'caption':'Max number of filter changes within a window of 10 minutes over the entire survey.'})
    order += 1
    m6 = configureMetric('MaxStateChangesWithinMetric',
                         kwargs={'changeCol':'filter', 'timespan':20},
                         displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                                      'caption':'Max number of filter changes within a window of 20 minutes over the entire survey.'})
    order += 1
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1, m2, m3, m4, m5, m6), constraints=[''], metadata='All visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)

    # Calculate some basic summary info about run, per filter, per proposal and for all proposals.
    propOrder = 0
    props = propids.keys() + ['All Props'] + ['WFD']
    for i, propid in enumerate(props):
        propOrder += 500
        order = propOrder
        for f in filters+['all']:
            if f != 'all':
                sqlconstraint = 'filter = "%s" and' %(f)
            else:
                sqlconstraint = ''
            if propid in WFDpropid:
                # Skip individual WFD propids (do in 'WFD')
                continue
            if propid == 'All Props':
                subgroup = 'All Props'
                sqlconstraint = sqlconstraint[:-4]
                metadata = '%s band, all props'%(f)
            elif propid == 'WFD':
                subgroup = 'WFD'
                sqlconstraint = sqlconstraint+' %s'%(wfdWhere)
                metadata = '%s band, WFD'%(f)
            else:
                subgroup = 'Per Prop'
                sqlconstraint = sqlconstraint+' propId=%d'%(propid)
                metadata = '%s band, %s'%(f, propids[propid])
            sqlconstraint = [sqlconstraint]
            metricList = []
            cols = ['finSeeing', 'filtSkyBrightness', 'airmass', 'fiveSigmaDepth', 'normairmass', 'dist2Moon']
            groups = [seeinggroup, skybrightgroup, airmassgroup, singlevisitdepthgroup, airmassgroup, dist2moongroup]
            for col, group in zip(cols, groups):
                metricList.append(configureMetric('MedianMetric', kwargs={'col':col},
                                                displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('MeanMetric', kwargs={'col':col},
                                                    displayDict={'group':group, 'subgroup':subgroup,
                                                                'order':order}))
                order += 1
                metricList.append(configureMetric('RmsMetric', kwargs={'col':col},
                                                    displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('NoutliersNsigmaMetric',
                                                    kwargs={'col':col, 'metricName':'N(-3Sigma) %s' %(col), 'nSigma':-3.},
                                                    displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('NoutliersNsigmaMetric',
                                                  kwargs={'col':col, 'metricName':'N(+3Sigma) %s' %(col), 'nSigma':3.},
                                                  displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('CountMetric', kwargs={'col':col, 'metricName':'Count %s' %(col)},
                                                  displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('PercentileMetric',
                                                    kwargs={'col':col, 'percentile':25},
                                                    displayDict={'group':group, 'subgroup':subgroup,
                                                                'order':order}))
                order += 1
                metricList.append(configureMetric('PercentileMetric',
                                                    kwargs={'col':col, 'percentile':50},
                                                    displayDict={'group':group, 'subgroup':subgroup,
                                                                'order':order}))
                order += 1
                metricList.append(configureMetric('PercentileMetric',
                                                    kwargs={'col':col, 'percentile':75},
                                                    displayDict={'group':group, 'subgroup':subgroup,
                                                                'order':order}))
                order += 1
            slicer = configureSlicer('UniSlicer', metricDict=makeDict(*metricList),
                                    constraints=sqlconstraint, metadata=metadata, metadataVerbatim=True)
            slicerList.append(slicer)


    # Calculate summary slew statistics.
    metricList = []
    # Mean Slewtime
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':1,
                                      'caption':'Mean slew time in seconds.'}))
    # Median Slewtime
    metricList.append(configureMetric('MedianMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':2,
                                      'caption':'Median slew time in seconds.'}))
    # Mean exposure time
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'visitExpTime'},
                                      displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':3,
                                                   'caption':'Mean visit on-sky time, in seconds.'}))
    # Mean visit time
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'visitTime'},
                                      displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':4,
                                                   'caption':
                                                   'Mean total visit time (including readout and shutter), in seconds.'}))
    metricDict = makeDict(*metricList)
    slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], metadata='All Visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)

    # Stats for angle:
    angles = ['telAlt', 'telAz', 'rotTelPos']

    order = 0
    for angle in angles:
        metricList = []
        metricList.append(configureMetric('MinMetric', kwargs={'col':angle, 'metricName':'Min'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                                                       'caption':'Minimum %s value, in radians.' %(angle)}))
        order += 1
        metricList.append(configureMetric('MaxMetric', kwargs={'col':angle, 'metricName':'Max'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                                                       'caption':'Maximum %s value, in radians.' %(angle)}))
        order += 1
        metricList.append(configureMetric('MeanMetric', kwargs={'col':angle, 'metricName':'Mean'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                                                       'caption':'Mean %s value, in radians.' %(angle)}))
        order += 1
        metricList.append(configureMetric('RmsMetric', kwargs={'col':angle, 'metricName':'RMS'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                                                       'caption':'Rms of %s value, in radians.' %(angle)}))
        order += 1
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], metadata=angle,
                                 metadataVerbatim=True, table='SlewState')
        slicerList.append(slicer)

    # Make some calls to other tables to get slew stats
    colDict = {'domAltSpd':'Dome Alt Speed','domAzSpd':'Dome Az Speed','telAltSpd': 'Tel Alt Speed',
               'telAzSpd': 'Tel Az Speed', 'rotSpd':'Rotation Speed'}
    order = 0
    for key in colDict:
        metricList=[]
        metricList.append(configureMetric('MaxMetric', kwargs={'col':key, 'metricName':'Max'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                                                       'caption':'Maximum slew speed for %s.' %(colDict[key])}))
        order += 1
        metricList.append(configureMetric('MeanMetric', kwargs={'col':key, 'metricName':'Mean'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                                                       'caption':'Mean slew speed for %s.' %(colDict[key])}))
        order += 1
        metricList.append(configureMetric('MaxPercentMetric', kwargs={'col':key, 'metricName':'% of slews'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                                                       'caption':'Percent of slews which are at maximum value of %s'
                                                       %(colDict[key])}))
        order += 1
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''],
                                 table='SlewMaxSpeeds', metadata=colDict[key], metadataVerbatim=True)
        slicerList.append(slicer)

    # Use the slew stats
    slewTypes = ['DomAlt', 'DomAz', 'TelAlt', 'TelAz', 'Rotator', 'Filter',
                 'TelOpticsOL', 'Readout', 'Settle', 'TelOpticsCL']

    order = 0
    for slewType in slewTypes:
        metricList = []
        metricList.append(configureMetric('CountRatioMetric',
                                          kwargs={'col':'actDelay', 'normVal':totalSlewN/100.0,
                                                  'metricName':'ActivePerc'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                                                       'caption':'Percent of total slews which include %s movement.'
                                                       %(slewType)}))

        order += 1
        metricList.append(configureMetric('MeanMetric',
                                          kwargs={'col':'actDelay',
                                                  'metricName':'ActiveAve'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                                                       'caption':'Mean amount of time (in seconds) for %s movements.'
                                                       %(slewType)}))
        order += 1
        metricList.append(configureMetric('MaxMetric',
                                          kwargs={'col':'actDelay',
                                                  'metricName':'Max'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                                                       'caption':'Max amount of time (in seconds) for %s movement.'
                                                       %(slewType)}))

        order += 1
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0 and activity="%s"'%slewType],
                                 table='SlewActivities', metadata=slewType,
                                 metadataVerbatim=True)
        slicerList.append(slicer)
        metricList = []
        metricList.append(configureMetric('CountRatioMetric',
                                          kwargs={'col':'actDelay', 'normVal':totalSlewN/100.0,
                                                  'metricName':'ActivePerc in crit'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                                                       'caption':'Percent of total slew which include %s movement, and are in critical path.' %(slewType)}))
        order += 1
        metricList.append(configureMetric('MeanMetric',
                                          kwargs={'col':'actDelay',
                                                  'metricName':'ActiveAve in crit'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                                                       'caption':'Mean time (in seconds) for %s movements, when in critical path.'
                                                       %(slewType)}))
        order += 1
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0 and inCriticalPath="True" and activity="%s"'%slewType],
                                 table='SlewActivities', metadata=slewType,
                                 metadataVerbatim=True)
        slicerList.append(slicer)
        metricList = []
        metricList.append(configureMetric('AveSlewFracMetric',
                                          kwargs={'col':'actDelay','activity':slewType,
                                                  'metricName':'Total Ave'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order}))
        order += 1
        metricList.append(configureMetric('SlewContributionMetric',
                                          kwargs={'col':'actDelay','activity':slewType,
                                                  'metricName':'Contribution'},
                                          displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order}))
        order += 1
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,constraints=[''],
                                 table='SlewActivities', metadata=slewType, metadataVerbatim=True)
        slicerList.append(slicer)

    # Count the number of visits per proposal, for all proposals, as well as the ratio of number of visits
    #  for each proposal compared to total number of visits.
    order = 1
    for propid in propids:
        sqlconstraint = ['propID = %s' %(propid)]
        m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'NVisits Per Proposal'},
                             summaryStats={'IdentityMetric':{'metricName':'Count'},
                                           'NormalizeMetric':{'normVal':totalNVisits, 'metricName':'Fraction of total'}},
                            displayDict={'group':summarygroup, 'subgroup':'1: NVisits', 'order':order,
                                         'caption':
                                         'Number of visits for %s proposal and fraction of total visits.'
                                         %(propids[propid])})
        order += 1
        slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1), constraints=sqlconstraint,
                                 metadata='%s' %(propids[propid]), metadataVerbatim=True)
        slicerList.append(slicer)
    # Count visits in WFD (as well as ratio of number of visits compared to total number of visits).
    sqlconstraint = ['%s' %(wfdWhere)]
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1),
                            constraints=sqlconstraint, metadata='WFD', metadataVerbatim=True)
    slicerList.append(slicer)
    # Count total number of visits.
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'TotalNVisits'},
                         summaryStats={'IdentityMetric':{'metricName':'Count'}},
                         displayDict={'group':summarygroup, 'subgroup':'1: NVisits', 'order':0})
    # Count total number of nights
    m2 = configureMetric('CountUniqueMetric', kwargs={'col':'night', 'metricName':'Nights with observations'},
                         summaryStats={'IdentityMetric':{'metricName':'(days)'}},
                         displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':1})
    m3 = configureMetric('FullRangeMetric', kwargs={'col':'night', 'metricName':'Total nights in survey'},
                         summaryStats={'ZeropointMetric':{'zp':1, 'metricName':'(days)'}},
                         displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':0})
    m4 = configureMetric('TeffMetric', kwargs={'metricName':'Total effective time of survey'},
                         summaryStats={'NormalizeMetric':{'normVal':24.0*60.0*60.0, 'metricName':'(days)'}},
                         displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':3})
    m5 = configureMetric('TeffMetric', kwargs={'metricName':'Normalized total effective time of survey', 'normed':True},
                         summaryStats={'IdentityMetric':{'metricName':'(fraction)'}},
                         displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':2})
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1, m2, m3, m4, m5), constraints=[''], metadata='All Visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)

    config.slicers=makeDict(*slicerList)
    return config
