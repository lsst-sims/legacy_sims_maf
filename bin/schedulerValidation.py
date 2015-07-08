#! /usr/bin/env python
import numpy as np
import os, sys, argparse
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plotters
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils
import healpy as hp
import matplotlib.pylab as plt


def makeBundleList(dbFile, nside=128, benchmark='design',
                   lonCol='fieldRA', latCol='fieldDec'):

    # List to hold everything we're going to make
    bundleList = []

    # Connect to the databse
    opsimdb = utils.connectOpsimDb(dbFile)

    # Fetch the proposal ID values from the database
    propids, propTags = opsimdb.fetchPropInfo()

    # Fetch the telescope location from config
    lat, lon, height = opsimdb.fetchLatLonHeight()

    commonname = ''.join([a for a in lonCol if a in latCol])
    if commonname == 'field':
        slicermetadata = ' (no dithers)'
    else:
        slicermetadata = ' (%s)' %(commonname)


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
    # Add variables to configure the slicer (in case we want to change it in the future).
    slicer = slicers.OpsimFieldSlicer()
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

            # Count the total number of visits.
            metric = metrics.CountMetric(col='expMJD', metricName = 'Nvisits')
            plotDict={'units':'Number of Visits', 'xMin':nvisitsMin,
                      'xMax':nvisitsMax, 'binsize':5}
            summaryStats=allStats
            displayDict={'group':nvisitgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Number of visits in filter %s, %s.' %(f, propCaption)}
            histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                       'binsize':5, 'xMin':nvisitsMin, 'xMax':nvisitsMax,
                       'legendloc':'upper right'}
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
                       'color':colors[f], 'label':'%s' %f, 'binsize':.02}
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
                           'units':'Number of Visits/Benchmark (%d)' %(benchmarkVals['nvisits'][f])},
                displayDict={'group':nvisitgroup, 'subgroup':'%s, ratio' %(subgroup),
                             'order':filtorder[f],
                             'caption': 'Number of visits in filter %s divided by %s value (%d), %s.'
                             %(f, benchmark, benchmarkVals['nvisits'][f], propCaption)},
                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s'%(f),
                           'xlabel':'Number of visits / benchmark',
                           'binsize':.05, 'xMin':0.475, 'xMax':1.525,
                           'legendloc':'upper right'}
                histNum += 1
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median individual visit five sigma limiting magnitude (individual image depth).
                metric= metrics.MedianMetric(col='fiveSigmaDepth')
                summaryStats=standardStats
                displayDict={'group':singlevisitdepthgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median single visit depth in filter %s, %s.' %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
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
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the median airmass.
                metric = metrics.MedianMetric(col='airmass')
                plotDict={'units':'X'}
                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median airmass in filter %s, %s.' %(f, propCaption)}
                # Calculate the median normalized airmass.
                metric = metrics.MedianMetric(col='normairmass')
                plotDict={'units':'X'}
                displayDict={'group':airmassgroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Median normalized airmass (airmass divided by the minimum airmass a field could reach) in filter %s, %s.'
                             %(f, propCaption)}
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
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
                bundle.histMerge = histMerge
                bundleList.append(bundle)
                # Calculate the mean of the hour angle.
                metric = metrics.MeanMetric(col='HA')
                plotDict={'xMin':-6, 'xMax':6}
                displayDict={'group':houranglegroup, 'subgroup':subgroup, 'order':filtorder[f],
                             'caption':'Full Range of the Hour Angle in filter %s, %s.'
                             %(f, propCaption)}))
                bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                    displayDict=displayDict, metadata=metadata,
                                                    summaryMetrics=summaryStats)
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
                           'cumulative':-1}
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
              'xMin':500, 'xMax':1500},
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
            plotDict={'units':'Number of Visits', 'plotMask':True, 'binsize':5},
            displayDict={'group':nvisitOpsimgroup, 'subgroup':'%s'%(propids[propid]),
                         'order':filtorder[f] + propOrder,
                         'caption':'Number of visits per opsim field in %s filter, for %s.'
                         %(f, propids[propid])}
            histMerge={'histNum':histNum, 'legendloc':'upper right', 'color':colors[f],
                       'label':'%s' %f, 'binsize':5}
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
                       'color':colors[f], 'label':'%s' %f, 'binsize':5}
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
        summaryStats={'TableFractionMetric':{}}
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
        metric = HourglassMetric(lat=lat*np.pi/180.,lon=lon*np.pi/180. , elev=height)
        displayDict={'group':hourglassgroup, 'subgroup':'Yearly', 'order':i}
        bundle = metricBundles.MetricBundle(metric, hourSlicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, metadata=metadata)
        bundleList.append(bundle)


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
            metric = metrics.CountMetric(col='fiveSigmaDepth', metricName='Single Visit Depth Histogram')
            histMerge={'histNum':histNum, 'legendloc':'upper right',
                       'color':colors[f], 'label':'%s'%f},
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
                       'color':colors[f], 'label':'%s'%f}
            displayDict={'group':skybrightgroup, 'subgroup':subgroup, 'order':filtorder[f],
                         'caption':'Histogram of the sky brightness in %s band, %s.' %(f, propCaption)}
            histNum += 1
            slicer = slicers.OneDSlicer(sliceColName='filtSkyBrightness', binsize=0.1,
                                        binMin=16, binMax=23}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, metadata=metadata,
                                            summaryMetrics=summaryStats)
            bundle.histMerge = histMerge
            bundleList.append(bundle)

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



    return bundleList

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the scheduler validation metrics')
    parser.add_argument('dbFile', type=str, default=None,help="full file path to the opsim sqlite file")

    parser.add_argument("--outDir",type=str, default='./Out', help='Output directory for MAF outputs.')
    parser.add_argument("--nside", type=int, default=128,
                        help="Resolution to run Healpix grid at (must be 2^x)")

    parser.add_argument('--benchmark', type=str, default='design',
                        help="Can be 'design' or 'requested'")

    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values and re-plot them.")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    bundleList = makeBundleList(args.dbFile,nside=args.nside, benchmark=args.benchmark,
                                plotOnly=args.plotOnly)

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

    for bdict in bundleDicts:
        group = metricBundles.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
        if args.plotOnly:
            # Load up the results
            pass
        else:
            group.runAll()
        group.plotAll()
        # Might consider killing bdict here to free up memory? Any bundles I want for later will
        # be persisted in the histBundleDict.

    for histNum in histBundleDict:
        # Need to plot up the merged histograms and write them to the resultsDb.
        ph = plots.PlotHandler(outDir=args.outDir, resultsDb=resultsDb)
        ph.setMetricBundles(histBundleDict[histNum])
        ph.plot(plotFunc=histBundleDict[histNum][0].histMerge['mergeFunc'])
