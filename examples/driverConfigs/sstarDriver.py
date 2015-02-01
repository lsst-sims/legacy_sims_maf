# A MAF config that replicates the SSTAR plots

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils
import numpy as np


def mConfig(config, runName, dbDir='.', outputDir='Out', slicerName='HealpixSlicer',
            benchmark='design', **kwargs):
    """
    A MAF config for SSTAR-like analysis of an opsim run.

    runName must correspond to the name of the opsim output
        (minus '_sqlite.db', although if added this will be stripped off)

    dbDir is the directory the database resides in

    outputDir is the output directory for MAF

    Uses 'slicerName' for metrics which have the option of using
      [HealpixSlicer, OpsimFieldSlicer, or HealpixSlicerDither]
      (dithered healpix slicer uses ditheredRA/dec values).

    Uses 'benchmark' (which can be design or stretch) to scale plots of number of visits and coadded depth.
    """

    # Setup Database access
    config.outputDir = outputDir
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
    if 'DD' not in propTags:
        propTags['DD'] = []
    if 'WFD' not in propTags:
        propTags['WFD'] = []
    DDpropid = propTags['DD']
    WFDpropid = propTags['WFD']

    # Fetch the telescope location from config
    lat,lon,height = opsimdb.fetchLatLonHeight()

    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
    wfdWhere = utils.createSQLWhere('WFD', propTags)
    print 'WFD "where" clause: %s' %(wfdWhere)
    ddWhere = utils.createSQLWhere('DD', propTags)
    print 'DD "where" clause: %s' %(ddWhere)

    # Fetch the total number of visits (to create fraction for number of visits per proposal)
    totalNVisits = opsimdb.fetchNVisits()
    totalSlewN = opsimdb.fetchTotalSlewN() 

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    # Set up benchmark values for Stretch and Design, scaled to length of opsim run.
    runLength = opsimdb.fetchRunLength()
    design, stretch = utils.scaleStretchDesign(runLength)

    # Set zeropoints and normalization values for plots (and range for nvisits plots).
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
    # make sure nvisitBench not zero
    for key in nvisitBench.keys():
        if nvisitBench[key] == 0:
            print 'Changing nvisit benchmark value to not be zero.'
            nvisitBench[key] = 1

    mag_DDzpoints = {'u':28.5, 'g':28.5, 'r':28.5, 'i':28.5, 'z':28.0, 'y':27.0}

    # Set range of values for visits plots.
    nVisits_plotRange = {'all':
                         {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200],
                          'z':[100, 250], 'y':[100,250]},
                         'DD':
                         {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],
                          'z':[7000, 10000], 'y':[5000, 8000]}}

    ####
    # Configure the most commonly used spatial slicers
    # Set slicer name and kwargs, so that we can vary these from the command line.
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
        slicermetadata = ' dithered'
    elif slicerName == 'OpsimFieldSlicer':
        slicerName = 'OpsimFieldSlicer'
        slicerkwargs = {}
        slicermetadata = ''
    else:
        raise ValueError('Do not understand slicerName %s: looking for one of %s' %(slicerName, slicerNames))
    print 'Using slicer %s for generic metrics over the sky.' %(slicerName)
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
                   'NoutliersNsigmaMetric 1':{'metricName':'p3Sigma', 'nSigma':3.},
                   'NoutliersNsigmaMetric 2':{'metricName':'m3Sigma', 'nSigma':-3.}}
    rangeStats={'PercentileMetric 1':{'metricName':'25th%ile', 'percentile':25},
                'PercentileMetric 2':{'metricName':'75th%ile', 'percentile':75},
                'MinMetric':{},
                'MaxMetric':{}}
    allStats = standardStats.copy()
    allStats.update(rangeStats)

    # Standardize a couple of labels (for ordering purposes in showMaf).
    summarygroup = 'A: Summary'
    srdgroup = 'B: SRD'
    nvisitgroup = 'C: NVisits'
    nvisitOpsimgroup = 'D: NVisits (opsim)'
    coaddeddepthgroup = 'E: Coadded depth'
    completenessgroup = 'F: Completeness'
    airmassgroup = 'G: Airmass'
    seeinggroup = 'H: Seeing'
    skybrightgroup = 'I: SkyBrightness'
    singlevisitdepthgroup = 'J: Single Visit Depth'
    hourglassgroup = 'K: Hourglass'
    slewgroup = 'L: Slew'

    ####
    # Start specifying metrics and slicers for MAF to run.

    slicerList=[]
    histNum = 0

    ## Metrics calculating values across the sky (healpix or opsim slicer).
    # Loop over a set of standard analysis metrics, for All Proposals, WFD only, and DD only. 

    startNum = histNum
    for i, prop in enumerate(['All Props', 'WFD', 'DD']):
        startNum += 100
        for f in filters:
            # Set some per-proposal information.
            if prop == 'All Props':
                subgroup = 'All Props'
                propCaption = ' for all proposals.'
                metadata = '%s band, all props' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s"' %(f)]
                nvisitsMin = nVisits_plotRange['all'][f][0]
                nvisitsMax = nVisits_plotRange['all'][f][1]
                mag_zp = mag_zpoints[f]
            elif prop == 'WFD':
                subgroup = 'WFD'
                propCaption = ' for all WFD proposals.'
                metadata = '%s band, WFD' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s" and %s' %(f, wfdWhere)]
                nvisitsMin = nVisits_plotRange['all'][f][0]
                nvisitsMax = nVisits_plotRange['all'][f][1]
                mag_zp = mag_zpoints[f]
            elif prop == 'DD':
                subgroup = 'DD'
                propCaption = ' for all DD proposals.'
                metadata = '%s band, DD' %(f) + slicermetadata
                sqlconstraint = ['filter = "%s" and %s' %(f, ddWhere)]
                nvisitsMin = nVisits_plotRange['DD'][f][0]
                nvisitsMax = nVisits_plotRange['DD'][f][1]
                mag_zp = mag_DDzpoints[f]
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
            # Count the number of visits as a ratio against a benchmark value, for 'all' and 'WFD'.
            if prop != 'DD':
                metricList.append(configureMetric('CountRatioMetric',
                                              kwargs={'col':'expMJD', 'normVal':nvisitBench[f],
                                                      'metricName':'NVisitsRatio'},
                                              plotDict={ 'binsize':0.05,'cbarFormat':'%2.2f',
                                                    'colorMin':0.5, 'colorMax':1.5, 'xMin':0.475, 'xMax':1.525,
                                                    'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])},
                                              displayDict={'group':nvisitgroup, 'subgroup':'%s, ratio' %(subgroup),
                                                           'order':filtorder[f],
                                                           'caption': 'Number of visits in filter %s divided by %s value (%d), %s.'
                                                     %(f, benchmark, nvisitBench[f], propCaption)},
                                              histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                                                         'xlabel':'Number of visits / benchmark',
                                                         'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                                                         'legendloc':'upper right'}))
                histNum += 1
                # Calculate the median individual visit five sigma limiting magnitude.
                metricList.append(configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'},
                                    summaryStats=standardStats,
                                    displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':'Median single visit depth in filter %s, %s.' %(f, propCaption)}))
                # Calculate the median individual visit sky brightness (normalized to a benchmark).
                metricList.append(configureMetric('MedianMetric',
                                                kwargs={'col':'filtSkyBrightness'},
                                                plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f]),
                                                        'xMin':-2, 'xMax':1},
                                                displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':
                                                'Median Sky Brightness in filter %s with expected zeropoint (%.2f) subtracted, %s. Fainter sky brightness values are more positive numbers.'
                                                %(f, sky_zpoints[f], propCaption)}))
                # Calculate the median delivered seeing.
                metricList.append(configureMetric('MedianMetric', kwargs={'col':'finSeeing'},
                                        plotDict={'normVal':seeing_norm[f],
                                                    'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])},
                                        displayDict={'group':seeinggroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                    'caption':
                                                    'Median Seeing in filter %s divided by expected value (%.2f), %s.'
                                                    %(f, seeing_norm[f], propCaption)}))
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
                                                            'caption':'Median normalized airmass in filter %s, %s.'
                                                            %(f, propCaption)}))
                # Calculate the maximum airmass.
                metricList.append(configureMetric('MaxMetric',
                                                kwargs={'col':'airmass'},
                                                plotDict={'units':'X'},
                                                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                                                'caption':'Max airmass in filter %s, %s.' %(f, propCaption)}))
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

    # Count the number of visits per filter for each proposal, over the sky.
    # Different from above, as uses opsim field slicer. Also, the min/max limits for these are allowed
    #  to float, so that we can really see what's going on in each proposal.
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
            slicer = configureSlicer('OpsimFieldSlicer',
                                     metricDict=metricDict,
                                     constraints=sqlconstraint,
                                     metadata='%s band, %s' %(f, propids[propid]),
                                     metadataVerbatim=True)
            slicerList.append(slicer)
        propOrder += 100
        histNum += 1

    # Run for combined WFD proposals if there's more than one.  Isn't this already being done above?--yes,
    # but possibly with the HealpixSlicer.
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
            slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict, constraints=sqlconstraint,
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
                            kwargs={'u':nvisitBench['u'], 'g':nvisitBench['g'], 'r':nvisitBench['r'],
                                    'i':nvisitBench['i'], 'z':nvisitBench['z'], 'y':nvisitBench['y']},
                            summaryStats={'TableFractionMetric':{}},
                            displayDict={'group':completenessgroup, 'subgroup':subgroup})
        metricDict = makeDict(m1)
        slicer = configureSlicer('OpsimFieldSlicer', metricDict=metricDict,
                                constraints=sqlconstraint, metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)


    # Calculate the fO metrics for all proposals and WFD only.
    fOnside = 64
    order = 0
    for prop in ('All Prop', 'WFD only'):
        if prop == 'All Prop':
            metadata = 'All proposals'
            sqlconstraint = ['']
        if prop == 'WFD only':
            metadata = 'WFD only'
            sqlconstraint = ['%s' %(wfdWhere)]
        # Configure the count metric which is what is used for f0 slicer.
        m1 = configureMetric('CountMetric',
                            kwargs={'col':'expMJD', 'metricName':'fO'},
                            plotDict={'units':'Number of Visits',
                                      'xMin':0,
                                      'xMax':1500},
                            summaryStats={'fOArea':{'nside':fOnside},
                                            'fONv':{'nside':fOnside}},
                            displayDict={'group':srdgroup, 'subgroup':'F0', 'displayOrder':order, 'caption':
                                        'FO metric: evaluates the overall efficiency of observing.'})
        order += 1
        slicer = configureSlicer('fOSlicer', kwargs={'nside':fOnside},
                                 metricDict=makeDict(m1), constraints=sqlconstraint,
                                 metadata=metadata, metadataVerbatim=True)
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

    # Histograms per filter for WFD only (generally used to produce merged histograms).
    startNum = histNum
    for f in filters:
        metadata = '%s band, WFD' %(f)
        # Reset histNum to starting value (to combine filters).
        histNum = startNum
        # Histogram the individual visit five sigma limiting magnitude.
        m1 = configureMetric('CountMetric',
                             kwargs={'col':'fiveSigmaDepth', 'metricName':'Single Visit Depth Histogram'},
                             histMerge={'histNum':histNum, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f},
                            displayDict={'group':singlevisitdepthgroup, 'subgroup':'WFD', 'order':filtorder[f],
                                         'caption':'Histogram of the single visit depth in %s band, WFD only.' %(f)})
        histNum += 1
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'fiveSigmaDepth', 'binsize':0.05},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s" %(f, wfdWhere)],
                                metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)
        # Histogram the individual visit sky brightness.
        m1 = configureMetric('CountMetric', kwargs={'col':'filtSkyBrightness', 'metricName':'Sky Brightness Histogram'},
                            histMerge={'histNum':histNum, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f},
                            displayDict={'group':skybrightgroup, 'subgroup':'WFD', 'order':filtorder[f],
                                         'caption':'Histogram of the sky brightness in %s band, WFD only.' %(f)})
        histNum += 1
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'filtSkyBrightness', 'binsize':0.1,
                                                       'binMin':16, 'binMax':23},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)],
                                metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)
        # Histogram the individual visit seeing.
        m1 = configureMetric('CountMetric', kwargs={'col':'finSeeing', 'metricName':'Seeing Histogram'},
                            histMerge={'histNum':histNum, 'legendloc':'upper right',
                                       'color':colors[f],'label':'%s'%f},
                            displayDict={'group':seeinggroup, 'subgroup':'WFD', 'order':filtorder[f],
                                         'caption':'Histogram of the seeing in %s band, WFD only.' %(f)} )
        histNum += 1
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'finSeeing', 'binsize':0.02},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)],
                                metadata=metadata, metadataVerbatim=True)
        slicerList.append(slicer)
        # Histogram the individual visit airmass values.
        m1 = configureMetric('CountMetric', kwargs={'col':'airmass', 'metricName':'Airmass Histogram'},
                             histMerge={'histNum':histNum, 'legendloc':'upper right',
                                       'color':colors[f], 'label':'%s'%f, 'xMin':1.0, 'xMax':2.0},
                            displayDict={'group':airmassgroup, 'subgroup':'WFD', 'order':filtorder[f],
                                         'caption':'Histogram of the airmass in %s band, WFD only.' %(f)})
        histNum += 1
        slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'airmass', 'binsize':0.01},
                                metricDict=makeDict(m1), constraints=["filter = '%s' and %s"%(f, wfdWhere)],
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
                                      'caption':'Open shutter fraction per night.'})
    m3 = configureMetric('NChangesMetric', kwargs={'col':'filter', 'metricName':'Filter Changes'},
                         summaryStats=allStats,
                         displayDict={'group':summarygroup, 'subgroup':'3: Obs Per Night',
                                     'caption':'Number of filter changes per night.'})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night','binsize':1},
                             metricDict=makeDict(m1, m2, m3),
                             constraints=[''], metadata='Per night', metadataVerbatim=True)
    slicerList.append(slicer)

    ## Unislicer (single number) metrics.

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
            cols = ['finSeeing', 'filtSkyBrightness', 'airmass', 'fiveSigmaDepth']
            groups = [seeinggroup, skybrightgroup, airmassgroup, singlevisitdepthgroup]
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
                                                    kwargs={'col':col, 'metricName':'m3Sigma %s' %(col), 'nSigma':-3.},
                                                    displayDict={'group':group, 'subgroup':subgroup, 'order':order}))
                order += 1
                metricList.append(configureMetric('NoutliersNsigmaMetric',
                                                  kwargs={'col':col, 'metricName':'p3Sigma %s' %(col), 'nSigma':3.},
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


    # Calculate SLEW statistics.
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
                                 metadataVerbatim=True, table='slewState')
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
                                 table='slewMaxSpeeds', metadata=colDict[key], metadataVerbatim=True)
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
                                 table='slewActivities', metadata=slewType,
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
                                 table='slewActivities', metadata=slewType,
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
                                 table='slewActivities', metadata=slewType, metadataVerbatim=True)
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
    m2 = configureMetric('UniqueMetric', kwargs={'col':'night', 'metricName':'Nights on sky'},
                                     displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time'})
    m3 = configureMetric('FullRangeMetric', kwargs={'col':'night', 'metricName':'Nights in survey'},
                         displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time'})
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1, m2, m3), constraints=[''], metadata='All Visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)

    config.slicers=makeDict(*slicerList)
    return config
