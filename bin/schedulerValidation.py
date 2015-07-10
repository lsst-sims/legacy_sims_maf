#! /usr/bin/env python
import numpy as np
import os, sys, argparse, copy
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plotters
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots
import lsst.sims.maf.utils as utils
import healpy as hp
import matplotlib.pylab as plt


def makeBundleList(dbFile, benchmark='design'):

    # List to hold everything we're going to make
    bundleList = []

    # Connect to the databse
    opsimdb = utils.connectOpsimDb(dbFile)

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
    slicer = slicers.OpsimFieldSlicer()
    slicermetadata = ''
    # For a few slicer/metric combos, we want to only create histograms (not skymaps or power spectra), but keep
    #  the rest of slicerkwargs.
    onlyHist = {'plotFuncs':'plotHistogram'}


    opsimHistPlot = plots.OpsimHistogram()

    ###
    # Configure some standard summary statistics dictionaries to apply to appropriate metrics.
    # Note there's a complication here, you can't configure multiple versions of a summary metric since that makes a
    # dict with repeated keys.  The workaround is to add blank space (or even more words) to one of
    # the keys, which will be stripped out of the metric class name when the object is instatiated.
    standardStats = [metrics.MeanMetric(),
                     metrics.RmsMetric(), metrics.MedianMetric(), metrics.CountMetric(),
                     metrics.NoutliersNsigmaMetric(metricName='N(+3Sigma)', nSigma=3),
                     metrics.NoutliersNsigmaMetric(metricName='N(-3Sigma)', nSigma=-3.)]

    rangeStats = [metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                  metrics.PercentileMetric(metricName='75th%ile', percentile=75),
                  metrics.MinMetric(), metrics.MaxMetric()]

    allStats = copy.deepcopy(standardStats)
    allStats.extend(rangeStats)

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
                sqlconstraint = 'filter = "%s"' %(f)
                nvisitsMin = nvisitsRange['all'][f][0]
                nvisitsMax = nvisitsRange['all'][f][1]
                mag_zp = benchmarkVals['coaddedDepth'][f]
            elif prop == 'WFD':
                subgroup = 'WFD'
                propCaption = ' for all WFD proposals'
                metadata = '%s band, WFD' %(f) + slicermetadata
                sqlconstraint = 'filter = "%s" and %s' %(f, wfdWhere)
                nvisitsMin = nvisitsRange['all'][f][0]
                nvisitsMax = nvisitsRange['all'][f][1]
                mag_zp = benchmarkVals['coaddedDepth'][f]
            elif prop == 'DD':
                if len(DDpropid) == 0:
                    continue
                subgroup = 'DD'
                propCaption = ' for all DD proposals'
                metadata = '%s band, DD' %(f) + slicermetadata
                sqlconstraint = 'filter = "%s" and %s' %(f, ddWhere)
                nvisitsMin = nvisitsRange['DD'][f][0]
                nvisitsMax = nvisitsRange['DD'][f][1]
                mag_zp = benchmarkDDVals['coaddedDepth'][f]
            # Reset histNum (for merged histograms, merged over all filters).
            histNum = startNum
            # Configure the metrics to run for this sql constraint (all proposals/wfd and filter combo).

            # Count the total number of visits.
            metric = metrics.CountMetric(col='expMJD', metricName = 'Nvisits')
            plotDict={'units':'Number of Visits', 'xMin':nvisitsMin,
                      'xMax':nvisitsMax, 'binsize':5}
            summaryStats=allStats
            displayDict={'group':nvisitgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Number of visits in filter %s, %s.' %(f, propCaption)}
            histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                       'binsize':5, 'xMin':nvisitsMin, 'xMax':nvisitsMax,
                       'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
            histNum += 1
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Calculate the coadded five sigma limiting magnitude (normalized to a benchmark).
            metric = metrics.Coaddm5Metric()
            plotDict={'zp':mag_zp, 'xMin':-0.6, 'xMax':0.6,
                      'units':'coadded m5 - %.1f' %mag_zp}
            summaryStats=allStats
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'binsize':.02, 'mergeFunc':opsimHistPlot}
            displayDict={'group':coaddeddepthgroup, 'subgroup':subgroup,
                         'order':filtorder[f],
                         'caption':
                         'Coadded depth in filter %s, with %s value subtracted (%.1f), %s. More positive numbers indicate fainter limiting magnitudes.' %(f, benchmark, mag_zp, propCaption)}
            histNum += 1
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)
            # Only calculate the rest of these metrics for NON-DD proposals.
            if prop != 'DD':
                # Count the number of visits as a ratio against a benchmark value, for 'all' and 'WFD'.
                metric = metrics.CountRatioMetric(col='expMJD', normVal=benchmarkVals['nvisits'][f],
                                                  metricName='NVisitsRatio')
                plotDict={ 'binsize':0.05,'cbarFormat':'%2.2f',
                           'colorMin':0.5, 'colorMax':1.5, 'xMin':0.475, 'xMax':1.525,
                           'units':'Number of Visits/Benchmark (%d)' %(benchmarkVals['nvisits'][f])}
                displayDict={'group':nvisitgroup, 'subgroup':'%s, ratio' %(subgroup),
                             'order':filtorder[f],
                             'caption': 'Number of visits in filter %s divided by %s value (%d), %s.'
                             %(f, benchmark, benchmarkVals['nvisits'][f], propCaption)}
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Number of visits / benchmark',
                           'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median individual visit five sigma limiting magnitude (individual image depth).
                metric= metrics.MedianMetric(col='fiveSigmaDepth')
                summaryStats=standardStats
                plotdict= {}
                displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median single visit depth in filter %s, %s.' %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Median 5-sigma depth (mags)',
                           'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median individual visit sky brightness (normalized to a benchmark).
                metric = metrics.MedianMetric(col='filtSkyBrightness')
                plotDict={'zp':benchmarkVals['skybrightness'][f],
                          'units':'Skybrightness - %.2f' %(benchmarkVals['skybrightness'][f]),
                          'xMin':-2, 'xMax':1}
                displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':
                             'Median Sky Brightness in filter %s with expected zeropoint (%.2f) subtracted, %s. Fainter sky brightness values are more positive numbers.'
                             %(f, benchmarkVals['skybrightness'][f], propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'binsize':.05, 'xMin':-2, 'xMax':2,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median delivered seeing.
                metric = metrics.MedianMetric(col='finSeeing')
                plotDict={'normVal':benchmarkVals['seeing'][f],
                          'units':'Median Seeing/(Expected seeing %.2f)'%(benchmarkVals['seeing'][f])}
                displayDict={'group':seeinggroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':
                             'Median Seeing in filter %s divided by expected value (%.2f), %s.'
                             %(f, benchmarkVals['seeing'][f], propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Seeing/benchmark seeing',
                           'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median airmass.
                metric = metrics.MedianMetric(col='airmass')
                plotDict={'units':'X'}
                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median airmass in filter %s, %s.' %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Median Airmass',
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                # Calculate the median normalized airmass.
                metric = metrics.MedianMetric(col='normairmass')
                plotDict={'units':'X'}
                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median normalized airmass (airmass divided by the minimum airmass a field could reach) in filter %s, %s.'
                             %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Median Normalized Airmass',
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the maximum airmass.
                metric = metrics.MaxMetric(col='airmass')
                plotDict={'units':'X'}
                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Max airmass in filter %s, %s.' %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Max Airmass',
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the mean of the hour angle.
                metric = metrics.MeanMetric(col='HA')
                plotDict={'xMin':-6, 'xMax':6}
                displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Full Range of the Hour Angle in filter %s, %s.'
                             %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Mean Hour Angle (Hours)',
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the Full Range of the hour angle.
                metric = metrics.FullRangeMetric(col='HA')
                plotDict={'xMin':0, 'xMax':12}
                displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Full Range of the Hour Angle in filter %s, %s.'
                             %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Full Hour Angle Range',
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the RMS of the position angle
                metric = metrics.RmsAngleMetric(col='rotSkyPos')
                plotDict={'xMin':0, 'xMax':float(np.pi)}
                displayDict={'group':rotatorgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'RMS of the position angle (angle between "up" in the camera and north on the sky) in filter %s, %s.'
                             %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'binsize':.05,
                           'legendloc':'upper right', 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle.histMerge = histMerge
                bundleList.append(bundle)


            # Tack on an extra copy of Nvisits with a cumulative histogram for WFD.
            if prop == 'WFD':
                metric = metrics.CountMetric(col='expMJD', metricName='Nvisits cumulative')
                plotDict={'units':'Number of Visits','xMin':0,
                          'xMax':nvisitsMax, 'binsize':5, 'cumulative':-1}
                displayDict={'group':nvisitgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Cumulative number of visits in filter %s, %s.'
                             %(f, propCaption)}
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'binsize':5, 'xMin':0, 'xMax':nvisitsMax, 'legendloc':'upper right',
                           'cumulative':-1, 'mergeFunc':opsimHistPlot}
                histNum += 1
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                bundle.histMerge = histMerge
                bundleList.append(bundle)


    # Count the number of visits in all filters together, WFD only.
    sqlconstraint = wfdWhere
    metadata='All filters WFD: histogram only'
    plotFunc = plots.OpsimHistogram()
    # Make the reverse cumulative histogram
    metric = metrics.CountMetric(col='expMJD', metricName='Nvisits, all filters, cumulative')
    plotDict={'units':'Number of Visits', 'binsize':5, 'cumulative':-1,
              'xMin':500, 'xMax':1500}
    displayDict={'group':nvisitgroup, 'subgroup':'WFD', 'order':0,
                 'caption':'Number of visits all filters, WFD only'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats, plotFuncs=[plotFunc])
    bundleList.append(bundle)
    # Regular Histogram
    metric = metrics.CountMetric(col='expMJD', metricName='Nvisits, all filters')
    plotDict={'units':'Number of Visits', 'binsize':5, 'cumulative':False,
              'xMin':500, 'xMax':1500}
    summaryStats=allStats
    displayDict={'group':nvisitgroup, 'subgroup':'WFD', 'order':0,
                 'caption':'Number of visits all filters, WFD only'}

    # Count the number of visits per filter for each individual proposal, over the sky.
    #  The min/max limits for these plots are allowed to float, so that we can really see what's going on in each proposal.
    propOrder = 0
    for propid in propids:
        for f in filters:
            # Count the number of visits.
            sqlconstraint = 'filter = "%s" and propID = %s' %(f,propid)
            metadata = '%s band, %s' %(f, propids[propid])
            metric = metrics.CountMetric(col='expMJD', metricName='NVisits Per Proposal')
            summaryStats=standardStats
            plotDict={'units':'Number of Visits', 'plotMask':True, 'binsize':5}
            displayDict={'group':nvisitOpsimgroup, 'subgroup':'%s'%(propids[propid]),
                         'order':filtorder[f] + propOrder,
                         'caption':'Number of visits per opsim field in %s filter, for %s.'
                         %(f, propids[propid])}
            histMerge={'histNum':histNum, 'legendloc':'upper right', 'color':colors[f],
                       'label':'%s' %f, 'binsize':5, 'mergeFunc':opsimHistPlot}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

        propOrder += 100
        histNum += 1

    # Run for combined WFD proposals if there's more than one. Similar to above, but with different nvisits limits.
    if len(WFDpropid) > 1:
        for f in filters:
            sqlconstraint = 'filter = "%s" and %s' %(f, wfdWhere)
            metadata='%s band, WFD' %(f)
            metric = metrics.CountMetric(col='expMJD', metricName='NVisits Per Proposal')
            summaryStats=standardStats
            plotDict={'units':'Number of Visits', 'binsize':5}
            displayDict={'group':nvisitOpsimgroup, 'subgroup':'WFD',
                         'order':filtorder[f] + propOrder,
                         'caption':'Number of visits per opsim field in %s filter, for WFD.' %(f)}
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'binsize':5, 'mergeFunc':opsimHistPlot}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)
        histNum += 1

    # Calculate the Completeness and Joint Completeness for all proposals and WFD only.
    for prop in ('All Props', 'WFD'):
        if prop == 'All Props':
            subgroup = 'All Props'
            metadata = 'All proposals'
            sqlconstraint = ''
            xlabel = '# visits (All Props) / (# WFD %s value)' %(benchmark)
        if prop == 'WFD':
            subgroup = 'WFD'
            metadata = 'WFD only'
            sqlconstraint = '%s' %(wfdWhere)
            xlabel = '# visits (WFD) / (# WFD %s value)' %(benchmark)
        # Configure completeness metric.
        metric = metrics.CompletenessMetric(u=benchmarkVals['nvisits']['u'],
                                            g=benchmarkVals['nvisits']['g'],
                                            r=benchmarkVals['nvisits']['r'],
                                            i=benchmarkVals['nvisits']['i'],
                                            z=benchmarkVals['nvisits']['z'],
                                            y=benchmarkVals['nvisits']['y'])
        plotDict={'xlabel':xlabel, 'units':xlabel, 'xMin':0.5, 'xMax':1.5, 'bins':50}
        summaryStats=[metrics.TableFractionMetric()]
        displayDict={'group':completenessgroup, 'subgroup':subgroup}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, metadata=metadata,
                                            summaryMetrics=summaryStats)
        bundleList.append(bundle)

    ## End of all-sky metrics.

    ## Hourglass metric.
    hourSlicer = slicers.HourglassSlicer()
    # Calculate Filter Hourglass plots per year (split to make labelling easier).
    yearDates = range(0,int(round(365*runLength))+365,365)
    for i in range(len(yearDates)-1):
        sqlconstraint = 'night > %i and night <= %i'%(yearDates[i],yearDates[i+1])
        metadata='Year %i-%i' %(i, i+1)
        metric = metrics.HourglassMetric(lat=lat*np.pi/180.,lon=lon*np.pi/180. , elev=height)
        displayDict={'group':hourglassgroup, 'subgroup':'Yearly', 'order':i}
        bundle = metricBundles.MetricBundle(metric, hourSlicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)


    ## Histograms of individual output values of Opsim. (one-d slicers).
    histFunc = plots.OneDBinnedData()
    # Histograms per filter for All & WFD only (generally used to produce merged histograms).
    startNum = histNum
    plotDict=None
    for i, prop in enumerate(['All Props', 'WFD']):
        for f in filters:
            # Set some per-proposal information.
            if prop == 'All Props':
                subgroup = 'All Props'
                propCaption = ' for all proposals.'
                metadata = '%s band, all props' %(f) + slicermetadata
                sqlconstraint = 'filter = "%s"' %(f)
                # Reset histNum to starting value (to combine filters).
                histNum = startNum
            elif prop == 'WFD':
                subgroup = 'WFD'
                propCaption = ' for all WFD proposals.'
                metadata = '%s band, WFD' %(f) + slicermetadata
                sqlconstraint = 'filter = "%s" and %s' %(f, wfdWhere)
                # Reset histNum to starting value (to combine filters).
                histNum = startNum + 20
            # Set up metrics and slicers for histograms.
            # Histogram the individual visit five sigma limiting magnitude (individual image depth).
            metric = metrics.CountMetric(col='fiveSigmaDepth', metricName='Single Visit Depth Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s'%f, 'mergeFunc':histFunc}
            displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the single visit depth in %s band, %s.' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='fiveSigmaDepth', binsize=0.05)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the individual visit sky brightness.
            metric = metrics.CountMetric(col='filtSkyBrightness', metricName='Sky Brightness Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s'%f, 'mergeFunc':histFunc}
            displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the sky brightness in %s band, %s.' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='filtSkyBrightness', binsize=0.1,
                                        binMin=16, binMax=23)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the individual visit seeing.
            metric = metrics.CountMetric(col='finSeeing', metricName='Seeing Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s'%f, 'mergeFunc':histFunc}
            displayDict={'group':seeinggroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the seeing in %s band, %s.' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='finSeeing', binsize=0.02)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the individual visit airmass values.
            metric = metrics.CountMetric(col='airmass', metricName='Airmass Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'xMin':1.0, 'xMax':2.0, 'mergeFunc':histFunc}
            displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the airmass in %s band, %s' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='airmass', binsize=0.01)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the individual visit normalized airmass values.
            metric = metrics.CountMetric(col='normairmass', metricName='Normalized Airmass Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'xMin':1.0, 'xMax':2.0, 'mergeFunc':histFunc}
            displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the normalized airmass in %s band, %s' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='normairmass', binsize=0.01)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)
            # Histogram the individual visit hour angle values.
            metric = metrics.CountMetric(col='HA', metricName='Hour Angle Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'xMin':-10., 'xMax':10, 'mergeFunc':histFunc}
            displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the hour angle in %s band, %s' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='HA', binsize=0.1)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the sky position angles (rotSkyPos)
            metric = metrics.CountMetric(col='rotSkyPos', metricName='Position Angle Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s' %f, 'xMin':0.,
                       'xMax':float(np.pi*2.), 'mergeFunc':histFunc}
            displayDict={'group':rotatorgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the position angle (in radians) in %s band, %s. The position angle is the angle between "up" in the image and North on the sky.' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='rotSkyPos', binsize=0.05)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

            # Histogram the individual visit distance to moon values.
            metric = metrics.CountMetric(col='dist2Moon', metricName='Distance to Moon Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s'%f,
                       'xMin':float(np.radians(15.)), 'xMax':float(np.radians(180.)),
                       'mergeFunc':histFunc}
            displayDict={'group':dist2moongroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the distance between the field and the moon (in radians) in %s band, %s' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='dist2Moon', binsize=0.05)
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, metadata=metadata,
                                                summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)


    # Slew histograms (time and distance).
    sqlconstraint = ''
    metric = metrics.CountMetric(col='slewTime', metricName='Slew Time Histogram')
    plotDict={'logScale':True, 'ylabel':'Count'}
    displayDict={'group':slewgroup, 'subgroup':'Slew Histograms',
                 'caption':'Histogram of slew times for all visits.'}
    slicer = slicers.OneDSlicer(sliceColName='slewTime', binsize=5)
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    metric = metrics.CountMetric(col='slewDist', metricName='Slew Distance Histogram')
    plotDict={'logScale':True, 'ylabel':'Count'}
    displayDict={'group':slewgroup, 'subgroup':'Slew Histograms',
                 'caption':'Histogram of slew distances for all visits.'}
    slicer = slicers.OneDSlicer(sliceColName='slewDist', binsize=0.05)
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict)
    bundleList.append(bundle)

    # Plots per night -- the number of visits and the open shutter time fraction.
    slicer = slicers.OneDSlicer(sliceColName='night',binsize=1)
    metadata = 'Per night'
    sqlconstraint = ''
    summaryStats=allStats

    metric = metrics.CountMetric(col='expMJD', metricName='NVisits')
    displayDict={'group':summarygroup, 'subgroup':'3: Obs Per Night',
                 'caption':'Number of visits per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)
    metric = metrics.OpenShutterFractionMetric()
    displayDict={'group':summarygroup, 'subgroup':'3: Obs Per Night',
                 'caption':'Open shutter fraction per night. This compares the on-sky image time against the on-sky time + slews/filter changes/readout, but does not include downtime due to weather.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.NChangesMetric(col='filter', metricName='Filter Changes')
    plotDict={'units':'Number of Filter Changes'}
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Number of filter changes per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.MinTimeBetweenStatesMetric(changeCol='filter')
    plotDict={'yMin':0, 'yMax':120}
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Minimum time between filter changes, in minutes.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=10)
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Number of filter changes, where the time between filter changes is shorter than 10 minutes, per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=20)
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Number of filter changes, where the time between filter changes is shorter than 20 minutes, per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=10)
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Max number of filter changes within a window of 10 minutes, per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=20)
    displayDict={'group':filtergroup, 'subgroup':'Per Night',
                 'caption':'Max number of filter changes within a window of 20 minutes, per night.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, metadata=metadata,
                                        summaryMetrics=summaryStats)
    bundleList.append(bundle)

    ## Unislicer (single number) metrics.
    slicer = slicers.UniSlicer()
    sqlcomstraint = ''
    metadata='All visits'
    order = 0

    metric = metrics.NChangesMetric(col='filter', metricName='Total Filter Changes')
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Total filter changes over survey'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    order += 1
    metric = metrics.MinTimeBetweenStatesMetric(changeCol='filter')
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Minimum time between filter changes, in minutes.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    order += 1
    metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=10)
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Number of filter changes faster than 10 minutes over the entire survey.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    order += 1
    metric = metrics.NStateChangesFasterThanMetric(changeCol='filter', cutoff=20)
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Number of filter changes faster than 20 minutes over the entire survey.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    order += 1
    metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=10)
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Max number of filter changes within a window of 10 minutes over the entire survey.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    order += 1
    metric = metrics.MaxStateChangesWithinMetric(changeCol='filter', timespan=20)
    displayDict={'group':filtergroup, 'subgroup':'Whole Survey', 'order':order,
                 'caption':'Max number of filter changes within a window of 20 minutes over the entire survey.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)
    order += 1

    # Calculate some basic summary info about run, per filter, per proposal and for all proposals.
    propOrder = 0
    props = propids.keys() + ['All Props'] + ['WFD']
    slicer = slicers.UniSlicer()
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

            cols = ['finSeeing', 'filtSkyBrightness', 'airmass', 'fiveSigmaDepth', 'normairmass', 'dist2Moon']
            groups = [seeinggroup, skybrightgroup, airmassgroup, singlevisitdepthgroup, airmassgroup, dist2moongroup]
            for col, group in zip(cols, groups):
                metric = metrics.MedianMetric(col=col)
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.MeanMetric(col=col)
                displayDict={'group':group, 'subgroup':subgroup,'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.RmsMetric(col=col)
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.NoutliersNsigmaMetric(col=col, metricName='N(-3Sigma) %s' %(col), nSigma=-3.)
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.NoutliersNsigmaMetric(col=col, metricName='N(+3Sigma) %s' %(col), nSigma=3.)
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.CountMetric(col=col, metricName='Count %s' %(col))
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.PercentileMetric(col=col, percentile=25)
                displayDict={'group':group, 'subgroup':subgroup,
                             'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.PercentileMetric(col=col, percentile=50)
                displayDict={'group':group, 'subgroup':subgroup,
                             'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1
                metric = metrics.PercentileMetric(col=col, percentile=75)
                displayDict={'group':group, 'subgroup':subgroup, 'order':order}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                                    displayDict=displayDict, metadata=metadata)
                bundleList.append(bundle)

                order += 1

    # Calculate summary slew statistics.
    slicer = slicers.UniSlicer()
    sqlconstraint = ''
    metadata='All Visits'
    # Mean Slewtime
    metric = metrics.MeanMetric(col='slewTime')
    displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':1,
                 'caption':'Mean slew time in seconds.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)
    # Median Slewtime
    metric = metrics.MedianMetric(col='slewTime')
    displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':2,
                 'caption':'Median slew time in seconds.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)
    # Mean exposure time
    metric = metrics.MeanMetric(col='visitExpTime')
    displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':3,
                 'caption':'Mean visit on-sky time, in seconds.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)
    # Mean visit time
    metric = metrics.MeanMetric(col='visitTime')
    displayDict={'group':slewgroup, 'subgroup':'Summary', 'order':4,
                 'caption':
                 'Mean total visit time (including readout and shutter), in seconds.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                        displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)


    # Stats for angle:
    angles = ['telAlt', 'telAz', 'rotTelPos']

    order = 0
    slicer = slicers.UniSlicer()
    sqlconstraint = ''
    slewStateBL = []
    for angle in angles:
        metadata=angle

        metric = metrics.MinMetric(col=angle, metricName='Min')
        displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                     'caption':'Minimum %s value, in radians.' %(angle)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewStateBL.append(bundle)

        order += 1
        metric = metrics.MaxMetric(col=angle, metricName='Max')
        displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                     'caption':'Maximum %s value, in radians.' %(angle)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewStateBL.append(bundle)

        order += 1
        metric = metrics.MeanMetric(col=angle, metricName='Mean')
        displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                     'caption':'Mean %s value, in radians.' %(angle)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewStateBL.append(bundle)

        order += 1
        metric = metrics.RmsMetric(col=angle, metricName='RMS')
        displayDict={'group':slewgroup, 'subgroup':'Slew Angles', 'order':order,
                     'caption':'Rms of %s value, in radians.' %(angle)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewStateBL.append(bundle)

        order += 1

    # Make some calls to other tables to get slew stats
    colDict = {'domAltSpd':'Dome Alt Speed','domAzSpd':'Dome Az Speed','telAltSpd': 'Tel Alt Speed',
               'telAzSpd': 'Tel Az Speed', 'rotSpd':'Rotation Speed'}
    order = 0
    slicer = slicers.UniSlicer()
    sqlconstraint = ''
    slewMaxSpeedsBL = []

    for key in colDict:
        metadata=colDict[key]
        metric = metrics.MaxMetric(col=key, metricName='Max')
        displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                     'caption':'Maximum slew speed for %s.' %(colDict[key])}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewMaxSpeedsBL.append(bundle)
        order += 1

        metric = metrics.MeanMetric(col=key, metricName='Mean')
        displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                     'caption':'Mean slew speed for %s.' %(colDict[key])}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewMaxSpeedsBL.append(bundle)

        order += 1
        metric = metrics.MaxPercentMetric(col=key, metricName='% of slews')
        displayDict={'group':slewgroup, 'subgroup':'Slew Speed', 'order':order,
                     'caption':'Percent of slews which are at maximum value of %s'
                     %(colDict[key])}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewMaxSpeedsBL.append(bundle)
        order += 1


    # Use the slew stats
    slewTypes = ['DomAlt', 'DomAz', 'TelAlt', 'TelAz', 'Rotator', 'Filter',
                 'TelOpticsOL', 'Readout', 'Settle', 'TelOpticsCL']

    order = 0
    sqlconstraint = ''
    slicer = slicers.UniSlicer()
    slewActivitiesBL = []

    for slewType in slewTypes:
        metadata=slewType

        metric = metrics.CountRatioMetric(col='actDelay', normVal=totalSlewN/100.0,
                                          metricName='ActivePerc')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                     'caption':'Percent of total slews which include %s movement.'
                     %(slewType)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1
        metric = metrics.MeanMetric(col='actDelay',metricName='ActiveAve')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                     'caption':'Mean amount of time (in seconds) for %s movements.'
                     %(slewType)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1
        metric = metrics.MaxMetric(col='actDelay', metricName='Max')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                     'caption':'Max amount of time (in seconds) for %s movement.'
                     %(slewType)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1

        sqlconstraint = 'actDelay>0 and inCriticalPath="True" and activity="%s"'%slewType
        metadata=slewType

        metric = metrics.CountRatioMetric(col='actDelay', normVal=totalSlewN/100.0,
                                          metricName='ActivePerc in crit')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                     'caption':'Percent of total slew which include %s movement, and are in critical path.' %(slewType)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1
        metric = metrics.MeanMetric(col='actDelay', metricName='ActiveAve in crit')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order,
                     'caption':'Mean time (in seconds) for %s movements, when in critical path.'
                     %(slewType)}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1

        sqlconstraint = ''
        metadata=slewType

        metric = metrics.AveSlewFracMetric(col='actDelay',activity=slewType, metricName='Total Ave')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)

        order += 1
        metric = metrics.SlewContributionMetric(col='actDelay',activity=slewType, metricName='Contribution')
        displayDict={'group':slewgroup, 'subgroup':'Slew Activity', 'order':order}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            displayDict=displayDict, metadata=metadata)
        slewActivitiesBL.append(bundle)
        order += 1

    # Count the number of visits per proposal, for all proposals, as well as the ratio of number of visits
    #  for each proposal compared to total number of visits.
    order = 1
    slicer = slicers.UniSlicer()
    for propid in propids:
        sqlconstraint = 'propID = %s' %(propid)
        metadata='%s' %(propids[propid])

        metric = metrics.CountMetric(col='expMJD', metricName='NVisits Per Proposal')
        summaryMetrics={'IdentityMetric':{'metricName':'Count'},
                      'NormalizeMetric':{'normVal':totalNVisits, 'metricName':'Fraction of total'}}
        displayDict={'group':summarygroup, 'subgroup':'1: NVisits', 'order':order,
                     'caption':
                     'Number of visits for %s proposal and fraction of total visits.'
                     %(propids[propid])}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            summaryMetrics=summaryStats,
                                            displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)
        order += 1

    # Count visits in WFD (as well as ratio of number of visits compared to total number of visits).
    sqlconstraint = '%s' %(wfdWhere)
    metadata='WFD'
    metric = metrics.CountMetric(col='expMJD', metricName='NVisits Per Proposal')
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                            summaryMetrics=summaryStats,
                                            displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)


    # Count total number of visits.
    sqlconstraint = ''
    slicer = slicers.UniSlicer()
    metadata='All Visits'


    metric = metrics.CountMetric(col='expMJD', metricName='TotalNVisits')
    summaryMetrics={'IdentityMetric':{'metricName':'Count'}},
    displayDict={'group':summarygroup, 'subgroup':'1: NVisits', 'order':0}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                         summaryMetrics=summaryStats,
                                         displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    # Count total number of nights
    metric = metrics.CountUniqueMetric(col='night', metricName='Nights with observations')
    summaryMetrics={'IdentityMetric':{'metricName':'(days)'}},
    displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':1}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                         summaryMetrics=summaryStats,
                                         displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    metric = metrics.FullRangeMetric(col='night', metricName='Total nights in survey')
    summaryMetrics={'ZeropointMetric':{'zp':1, 'metricName':'(days)'}},
    displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':0}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                         summaryMetrics=summaryStats,
                                         displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    metric = metrics.TeffMetric(metricName='Total effective time of survey')
    summaryMetrics={'NormalizeMetric':{'normVal':24.0*60.0*60.0, 'metricName':'(days)'}}
    displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':3}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                         summaryMetrics=summaryStats,
                                         displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)

    metric = metrics.TeffMetric(metricName='Normalized total effective time of survey', normed=True)
    summaryMetrics={'IdentityMetric':{'metricName':'(fraction)'}}
    displayDict={'group':summarygroup, 'subgroup':'2: On-sky Time', 'order':2}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint,
                                         summaryMetrics=summaryStats,
                                         displayDict=displayDict, metadata=metadata)
    bundleList.append(bundle)


    # Check the Alt-Az pointing history
    slicer = slicers.HealpixSlicer(nside=64, latCol='altitude', lonCol='azimuth', useCache=False)
    metric = metrics.CountMetric('expMJD')
    plotDict = {'rot':(0,90,0)}
    plotFunc = plots.HealpixSkyMap()
    for f in filters:
        sqlconstraint = 'filter = "%s"' %(f)
        displayDict={'group':'AltAzPointing',  'order':filtorder[f],
                     'caption':
                     'Pointing History on the alt-az sky (zenith center)'}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            plotFuncs=[plotFunc], displayDict=displayDict)
        bundleList.append(bundle)
    displayDict={'group':'AltAzPointing','subgroup':'All Filters'}
    bundle = metricBundles.MetricBundle(metric, slicer, ''
                                        , plotDict=plotDict,
                                        plotFuncs=[plotFunc])
    bundleList.append(bundle)

    return bundleList, slewStateBL, slewMaxSpeedsBL, slewActivitiesBL

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the scheduler validation metrics')
    parser.add_argument('dbFile', type=str, default=None,help="full file path to the opsim sqlite file")

    parser.add_argument("--outDir",type=str, default='./Out', help='Output directory for MAF outputs.')

    parser.add_argument('--benchmark', type=str, default='design',
                        help="Can be 'design' or 'requested'")

    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values and re-plot them.")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    bundleList, slewStateBL, slewMaxSpeedsBL, slewActivitiesBL = makeBundleList(args.dbFile,
                                                                                benchmark=args.benchmark)

    # Make a dictionary with all the bundles that need to be histogram merged
    histNums = []
    for bundle in bundleList:
        if hasattr(bundle, 'histMerge'):
            histNums.append(bundle.histMerge['histNum'])
    histNums = list(set(histNums))
    histBundleDict={}
    for num in histNums:
        histBundleDict[num] = []
    for bundle in bundleList:
        if hasattr(bundle, 'histMerge'):
            histBundleDict[bundle.histMerge['histNum']].append(bundle)

    bundleDicts = utils.bundleList2Dicts(bundleList)
    resultsDb = db.ResultsDb(outDir=args.outDir)
    opsdb = utils.connectOpsimDb(args.dbFile)

    # Do the ones that need a different table
    for bundleL,table in zip( [slewStateBL, slewMaxSpeedsBL, slewActivitiesBL ],['SlewState', 'SlewMaxSpeeds','SlewActivities']):
        bds = utils.bundleList2Dicts(bundleL)
        for bd in bds:
            group = metricBundles.MetricBundleGroup(bd, opsdb, outDir=args.outDir,
                                                    resultsDb=resultsDb, dbTable=table)
            group.runAll()
            group.plotAll()


    # Can we do this loop in parallel? I _really_ want to do this in parallel.
    for bdict in bundleDicts:
        group = metricBundles.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
        if args.plotOnly:
            # Load up the results
            pass
        else:
            group.runAll()
        group.plotAll()
        plt.close('all')
        # Might consider killing bdict here to free up memory? Any bundles I want for later will
        # be persisted in the histBundleDict.

    for histNum in histBundleDict:
        # Need to plot up the merged histograms and write them to the resultsDb.
        ph = plots.PlotHandler(outDir=args.outDir, resultsDb=resultsDb)
        ph.setMetricBundles(histBundleDict[histNum])
        ph.setPlotDict(plotDict=histBundleDict[histNum][0].histMerge,
                       plotFunc=histBundleDict[histNum][0].histMerge['mergeFunc'])
        ph.plot(histBundleDict[histNum][0].histMerge['mergeFunc'])
        plt.close('all')


    utils.writeConfigs(opsdb, args.outDir)
