#! /usr/bin/env python
import os, sys, argparse
import numpy as np
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import healpy as hp
import warnings

import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.utils as utils


def makeBundleList(dbFile, runName=None, nside=64, benchmark='design',
                   lonCol='fieldRA', latCol='fieldDec', seeingCol='FWHMgeom'):
    """
    make a list of metricBundle objects to look at the scientific performance
    of an opsim run.
    """

    # List to hold everything we're going to make
    bundleList = []

    # List to hold metrics that shouldn't be saved
    noSaveBundleList = []

    # Connect to the databse
    opsimdb = utils.connectOpsimDb(dbFile)
    if runName is None:
        runName = os.path.basename(dbFile).replace('_sqlite.db', '')

    # Fetch the proposal ID values from the database
    propids, propTags = opsimdb.fetchPropInfo()

    # Fetch the telescope location from config
    lat, lon, height = opsimdb.fetchLatLonHeight()

    # Add metadata regarding dithering/non-dithered.
    commonname = ''.join([a for a in lonCol if a in latCol])
    if commonname == 'field':
        slicermetadata = ' (non-dithered)'
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
    colors={'u':'cyan','g':'g','r':'y','i':'r','z':'m', 'y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    # Easy way to run through all fi

    # Set up a list of common summary stats
    commonSummary = [metrics.MeanMetric(), metrics.RobustRmsMetric(), metrics.MedianMetric(),
                     metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                     metrics.PercentileMetric(metricName='75th%ile', percentile=75),
                     metrics.MinMetric(), metrics.MaxMetric()]
    allStats = commonSummary

    # Set up some 'group' labels
    reqgroup = 'A: Required SRD metrics'
    depthgroup = 'B: Depth per filter'
    uniformitygroup = 'C: Uniformity'
    seeinggroup = 'D: Seeing distribution'
    transgroup = 'E: Transients'
    sngroup = 'F: SN Ia'
    altAzGroup = 'G: Alt Az'
    rangeGroup = 'H: Range of Dates'
    intergroup = 'I: Inter-Night'
    phaseGroup = 'J: Max Phase Gap'
    NEOGroup = 'K: NEO Detection'

    # Set up an object to track the metricBundles that we want to combine into merged plots.
    mergedHistDict = {}

    # Set the histogram merge function.
    mergeFunc = plots.HealpixHistogram()

    keys = ['NVisits', 'coaddm5', 'NormEffTime', 'Minseeing', 'seeingAboveLimit', 'minAirmass',
            'fracAboveAirmass']

    for key in keys:
        mergedHistDict[key] = plots.PlotBundle(plotFunc=mergeFunc)

    ##
    # Calculate the fO metrics for all proposals and WFD only.
    order = 0
    for prop in ('All prop', 'WFD only'):
        if prop == 'All prop':
            metadata = 'All Visits' + slicermetadata
            sqlconstraint = ''
        if prop == 'WFD only':
            metadata = 'WFD only' + slicermetadata
            sqlconstraint = '%s' %(wfdWhere)
        # Configure the count metric which is what is used for f0 slicer.
        m1 = metrics.CountMetric(col='expMJD', metricName='fO')
        plotDict={'xlabel':'Number of Visits','Asky':benchmarkVals['Area'],
                  'Nvisit':benchmarkVals['nvisitsTotal'],
                  'xMin':0, 'xMax':1500}
        summaryMetrics=[metrics.fOArea(nside=nside, norm=False, metricName='fOArea: Nvisits (#)',
                                       Asky=benchmarkVals['Area'], Nvisit=benchmarkVals['nvisitsTotal']),
                        metrics.fOArea(nside=nside, norm=True, metricName='fOArea: Nvisits/benchmark',
                                       Asky=benchmarkVals['Area'], Nvisit=benchmarkVals['nvisitsTotal']),
                        metrics.fONv(nside=nside, norm=False, metricName='fONv: Area (sqdeg)',
                                     Asky=benchmarkVals['Area'], Nvisit=benchmarkVals['nvisitsTotal']),
                        metrics.fONv(nside=nside, norm=True, metricName='fONv: Area/benchmark',
                                     Asky=benchmarkVals['Area'], Nvisit=benchmarkVals['nvisitsTotal'])]
        caption = 'The FO metric evaluates the overall efficiency of observing. '
        caption += 'fOArea: Nvisits = %.1f sq degrees receive at least this many visits out of %d. ' %(benchmarkVals['Area'], benchmarkVals['nvisitsTotal'])
        caption += 'fONv: Area = this many square degrees out of %.1f receive at least %d visits.' %(benchmarkVals['Area'], benchmarkVals['nvisitsTotal'])
        displayDict={'group':reqgroup, 'subgroup':'F0', 'displayOrder':order, 'caption':caption}
        order += 1
        slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)

        bundle = metricBundles.MetricBundle(m1, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, summaryMetrics=summaryMetrics,
                                            plotFuncs=[plots.FOPlot()],
                                            runName=runName, metadata=metadata)
        bundleList.append(bundle)

    ###
    # Calculate the Rapid Revisit Metrics.
    order = 0
    metadata = 'All Visits' + slicermetadata
    sqlconstraint = ''
    dTmin = 40.0 # seconds
    dTmax = 30.0 # minutes
    minNvisit = 100
    pixArea = float(hp.nside2pixarea(nside, degrees=True))
    scale = pixArea * hp.nside2npix(nside)
    cutoff1 = 0.15
    extraStats1 = [metrics.FracBelowMetric(cutoff=cutoff1, scale=scale, metricName='Area (sq deg)')]
    extraStats1.extend(commonSummary)
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    m1 = metrics.RapidRevisitMetric(metricName='RapidRevisitUniformity',
                                    dTmin=dTmin/60.0/60.0/24.0, dTmax=dTmax/60.0/24.0,
                                    minNvisits=minNvisit)

    plotDict={'xMin':0, 'xMax':1}
    summaryStats=extraStats1
    caption = 'Deviation from uniformity for short revisit timescales, between %s and %s seconds, ' %(dTmin, dTmax)
    caption += 'for pointings with at least %d visits in this time range. ' %(minNvisit)
    caption += 'Summary statistic "Area" below indicates the area on the sky which has a deviation from uniformity of < %.2f.' %(cutoff1)
    displayDict = {'group':reqgroup, 'subgroup':'Rapid Revisit', 'displayOrder':order, 'caption':caption}
    bundle = metricBundles.MetricBundle(m1, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1

    m2 = metrics.NRevisitsMetric(dT=dTmax)
    plotDict={'xMin':0, 'xMax':1000, 'logScale':True}
    cutoff2 = 800
    extraStats2 = [metrics.FracAboveMetric(cutoff=cutoff2, scale=scale, metricName='Area (sq deg)')]
    extraStats2.extend(commonSummary)
    caption = 'Number of consecutive visits with return times faster than %.1f minutes, in any filter, all proposals. ' %(dTmax)
    caption += 'Summary statistic "Area" below indicates the area on the sky which has more than %d revisits within this time window.' %(cutoff2)
    summaryStats= extraStats2
    displayDict = {'group':reqgroup, 'subgroup':'Rapid Revisit', 'displayOrder':order, 'caption':caption}
    bundle = metricBundles.MetricBundle(m2, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    m3 = metrics.NRevisitsMetric(dT=dTmax, normed=True)
    plotDict={'xMin':0, 'xMax':1, 'cbarFormat':'%.1f'}
    cutoff3 = 0.6
    extraStats3 = [metrics.FracAboveMetric(cutoff=cutoff3, scale=scale, metricName='Area (sq deg)')]
    extraStats3.extend(commonSummary)
    summaryStats= extraStats3
    caption = 'Fraction of total visits where consecutive visits have return times faster than %.1f minutes, in any filter, all proposals. ' %(dTmax)
    caption += 'Summary statistic "Area" below indicates the area on the sky which has more than %d revisits within this time window.' %(cutoff3)
    displayDict = {'group':reqgroup, 'subgroup':'Rapid Revisit', 'displayOrder':order, 'caption':caption}
    bundle = metricBundles.MetricBundle(m3, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1


    # And add a histogram of the time between quick revisits.
    binMin = 0
    binMax = 120.
    binsize= 3.
    bins = np.arange(binMin/60.0/24.0, (binMax+binsize)/60./24., binsize/60./24.)
    m1 = metrics.TgapsMetric(bins=bins, metricName='dT visits')
    plotDict={'bins':bins, 'xlabel':'dT (minutes)'}
    displayDict={'group':reqgroup, 'subgroup':'Rapid Revisit', 'order':order,
                 'caption':'Histogram of the time between consecutive revisits (<%.1f minutes), over entire sky.' %(binMax)}
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    plotFunc = plots.SummaryHistogram()
    bundle = metricBundles.MetricBundle(m1, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName, metadata=metadata, plotFuncs=[plotFunc])
    bundleList.append(bundle)
    order += 1


    ##
    # Trigonometric parallax and proper motion @ r=20 and r=24
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    sqlconstraint = ''
    order = 0
    metric = metrics.ParallaxMetric(metricName='Parallax 20', rmag=20, seeingCol=seeingCol)
    summaryStats=allStats
    plotDict={'cbarFormat':'%.1f', 'xMin':0, 'xMax':3}
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption':'Parallax precision at r=20. (without refraction).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric=metrics.ParallaxMetric(metricName='Parallax 24', rmag=24, seeingCol=seeingCol)
    plotDict={'cbarFormat':'%.1f', 'xMin':0, 'xMax':10}
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption':'Parallax precision at r=24. (without refraction).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric=metrics.ParallaxMetric(metricName='Parallax Normed', rmag=24, normalize=True,
                                  seeingCol=seeingCol)
    plotDict={'xMin':0.5, 'xMax':1.0}
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption':
                 'Normalized parallax (normalized to optimum observation cadence, 1=optimal).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage 20',rmag=20, seeingCol=seeingCol)
    plotDict={}
    caption = """Parallax factor coverage for an r=20 star (0 is bad, 0.5-1 is good). One expects the parallax factor coverage to vary because stars on the ecliptic can be observed when they have no parallax offset while stars at the pole are always offset by the full parallax offset."""
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxCoverageMetric(metricName='Parallax Coverage 24',rmag=24, seeingCol=seeingCol)
    plotDict={}
    caption = """Parallax factor coverage for an r=24 star (0 is bad, 0.5-1 is good). One expects the parallax factor coverage to vary because stars on the ecliptic can be observed when they have no parallax offset while stars at the pole are always offset by the full parallax offset."""
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption': caption}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxHADegenMetric(metricName='Parallax-DCR degeneracy 20',rmag=20,
                                           seeingCol=seeingCol)
    plotDict={}
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption':
                 'Correlation between parallax offset magnitude and hour angle an r=20 star (0 is good, near -1 or 1 is bad).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric = metrics.ParallaxHADegenMetric(metricName='Parallax-DCR degeneracy 24',rmag=24,
                                           seeingCol=seeingCol)
    plotDict={}
    displayDict={'group':reqgroup, 'subgroup':'Parallax', 'order':order,
                 'caption':
                 'Correlation between parallax offset magnitude and hour angle an r=24 star (0 is good, near -1 or 1 is bad).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1

    metric=metrics.ProperMotionMetric(metricName='Proper Motion 20', rmag=20, seeingCol=seeingCol)

    summaryStats=allStats
    plotDict={'xMin':0, 'xMax':3}
    displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                 'caption':'Proper Motion precision at r=20.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric=metrics.ProperMotionMetric(rmag=24, metricName='Proper Motion 24', seeingCol=seeingCol)
    summaryStats=allStats
    plotDict={'xMin':0, 'xMax':10}
    displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                 'caption':'Proper Motion precision at r=24.'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1
    metric=metrics.ProperMotionMetric(rmag=24,normalize=True, metricName='Proper Motion Normed',
                                      seeingCol=seeingCol)
    plotDict={'xMin':0.2, 'xMax':0.7}
    displayDict={'group':reqgroup, 'subgroup':'Proper Motion', 'order':order,
                 'caption':'Normalized proper motion at r=24 (normalized to optimum observation cadence - start/end. 1=optimal).'}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, summaryMetrics=summaryStats,
                                        runName=runName, metadata=metadata)
    bundleList.append(bundle)
    order += 1

    ##
    # Calculate the time uniformity in each filter, for each year.
    order = 0

    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    plotFuncs = [plots.TwoDMap()]
    step = 0.5
    bins = np.arange(0,365.25*10+40,40)-step
    metric = metrics.AccumulateUniformityMetric(bins=bins)
    plotDict={'xlabel':'Night (days)', 'xextent':[bins.min()+step,bins.max()+step], 'cbarTitle':'Uniformity'}
    for f in filters:
        sqlconstraint = 'filter = "%s"' %(f)
        caption = 'Deviation from uniformity in %s band. Northern Healpixels are at the top of the image.' %(f)
        caption += '(0=perfectly uniform, 1=perfectly nonuniform).'
        displayDict = {'group':uniformitygroup, 'subgroup':'per night',
                       'displayOrder':filtorder[f], 'caption': caption}
        metadata = '%s band' %(f) + slicermetadata
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            plotFuncs=plotFuncs)
        noSaveBundleList.append(bundle)

    ##
    # Depth metrics.
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    for f in filters:
        propCaption = '%s band, all proposals %s' %(f, slicermetadata)
        sqlconstraint = 'filter = "%s"' %(f)
        metadata = '%s band' %(f) + slicermetadata
        # Number of visits.
        metric = metrics.CountMetric(col='expMJD', metricName='NVisits')
        plotDict={'xlabel':'Number of visits',
                  'xMin':nvisitsRange['all'][f][0],
                  'xMax':nvisitsRange['all'][f][1], 'binsize':5,
                  'logScale':True, 'nTicks':4, 'colorMin':1}
        summaryStats=allStats
        displayDict={'group':depthgroup, 'subgroup':'Nvisits', 'order':filtorder[f],
                     'caption':'Number of visits in filter %s, %s.' %(f, propCaption)}
        histMerge={'color':colors[f], 'label':'%s'%(f),
                   'binsize':5,
                   'xMin':nvisitsRange['all'][f][0], 'xMax':nvisitsRange['all'][f][1],
                   'legendloc':'upper right'}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            summaryMetrics=summaryStats)
        mergedHistDict['NVisits'].addBundle(bundle,plotDict=histMerge)
        bundleList.append(bundle)
        # Coadded depth.
        metric = metrics.Coaddm5Metric()
        plotDict={'zp':benchmarkVals['coaddedDepth'][f], 'xMin':-0.8, 'xMax':0.8,
                  'xlabel':'coadded m5 - %.1f' %benchmarkVals['coaddedDepth'][f]}
        summaryStats=allStats
        histMerge={'legendloc':'upper right', 'color':colors[f], 'label':'%s' %f, 'binsize':.02,
                   'xlabel':'coadded m5 - benchmark value'}
        caption = 'Coadded depth in filter %s, with %s value subtracted (%.1f), %s. More positive numbers indicate fainter limiting magnitudes.'\
            %(f, benchmark, benchmarkVals['coaddedDepth'][f], propCaption)
        displayDict={'group':depthgroup, 'subgroup':'Coadded Depth',
                     'order':filtorder[f],'caption':caption}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName,  metadata=metadata,
                                            summaryMetrics=summaryStats)
        mergedHistDict['coaddm5'].addBundle(bundle,plotDict=histMerge)
        bundleList.append(bundle)
        # Effective time.
        metric = metrics.TeffMetric(metricName='Normalized Effective Time',normed=True,
                                    fiducialDepth= benchmarkVals['singleVisitDepth'])
        plotDict={'xMin':0.1, 'xMax':1.1}
        summaryStats=allStats
        histMerge={'legendLoc':'upper right', 'color':colors[f], 'label':'%s' %f, 'binsize':0.02}
        caption = '"Time Effective" in filter %s, calculated with fiducial single-visit depth of %s mag. '%(f, benchmarkVals['singleVisitDepth'][f])
        caption += 'Normalized by the fiducial time effective, if every observation was at the fiducial depth.'
        displayDict={'group':depthgroup, 'subgroup':'Time Eff.', 'order':filtorder[f], 'caption':caption}
        bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            summaryMetrics=summaryStats)
        mergedHistDict['NormEffTime'].addBundle(bundle,plotDict=histMerge)
        bundleList.append(bundle)


    # Put in a z=0.5 Type Ia SN, based on Cambridge 2015 workshop notebook.
    # Check for 1) detection in any band, 2) detection on the rise in any band,
    # 3) good characterization
    peaks = {'uPeak':25.9, 'gPeak':23.6, 'rPeak':22.6, 'iPeak':22.7, 'zPeak':22.7,'yPeak':22.8}
    peakTime = 15.
    transDuration = peakTime+30. # Days
    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          metricName='SNDetection',**peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected in any filter'
    displayDict={'group':transgroup,  'subgroup':'Detected', 'caption':caption}
    sqlconstraint = ''
    metadata = '' + slicermetadata
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName, metadata=metadata)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          nPrePeak=1, metricName='SNAlert', **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected pre-peak in any filter'
    displayDict={'group':transgroup,  'subgroup':'Detected on the rise', 'caption':caption}
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName, metadata=metadata)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.,
                                     transDuration=transDuration, peakTime=peakTime,
                                     surveyDuration=runLength, metricName='SNLots',
                                     nFilters=3, nPrePeak=3, nPerLC=2, **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are observed 6 times, 3 pre-peak, 3 post-peak, with observations in 3 filters'
    displayDict={'group':transgroup,  'subgroup':'Well observed', 'caption':caption}
    sqlconstraint = 'filter="r" or filter="g" or filter="i" or filter="z" '
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName, metadata=metadata)
    bundleList.append(bundle)


    # Good seeing in r/i band metrics, including in first/second years.
    order = 0
    for tcolor, tlabel, timespan in zip(['k', 'g', 'r'], ['10 years', '1 year', '2 years'],
                                        ['', ' and night<=365', ' and night<=730']):
        order += 1
        for f in (['r', 'i']):
            sqlconstraint = 'filter = "%s" %s' %(f, timespan)
            propCaption = '%s band, all proposals %s, over %s.' %(f, slicermetadata, tlabel)
            metadata = '%s band, %s' %(f, tlabel) + slicermetadata
            seeing_limit = 0.7
            airmass_limit = 1.2
            metric = metrics.MinMetric(col=seeingCol)
            summaryStats=allStats
            plotDict={'xMin':0.35, 'xMax':0.9, 'color':tcolor}
            displayDict={'group':seeinggroup, 'subgroup':'Best Seeing',
                         'order':filtorder[f]*100+order,
                         'caption':'Minimum FWHMgeom values in %s.' %(propCaption)}
            histMerge={'label':'%s %s' %(f, tlabel), 'color':tcolor,
                       'binsize':0.03, 'xMin':0.35, 'xMax':0.9, 'legendloc':'upper right'}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            summaryMetrics=summaryStats)
            mergedHistDict['Minseeing'].addBundle(bundle,plotDict=histMerge)
            bundleList.append(bundle)

            metric = metrics.FracAboveMetric(col=seeingCol, cutoff = seeing_limit)
            summaryStats=allStats
            plotDict={'xMin':0, 'xMax':1, 'color':tcolor}
            displayDict={'group':seeinggroup, 'subgroup':'Good seeing fraction',
                         'order':filtorder[f]*100+order,
                         'caption':'Fraction of total images with FWHMgeom worse than %.1f, in %s'
                         %(seeing_limit, propCaption)}
            histMerge={'color':tcolor, 'label':'%s %s' %(f, tlabel),
                       'binsize':0.05, 'legendloc':'upper right'}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            summaryMetrics=summaryStats)
            mergedHistDict['seeingAboveLimit'].addBundle(bundle,plotDict=histMerge)
            bundleList.append(bundle)

            metric = metrics.MinMetric(col='airmass')
            plotDict={'xMin':1, 'xMax':1.5, 'color':tcolor}
            summaryStats=allStats
            displayDict={'group':seeinggroup, 'subgroup':'Best Airmass',
                         'order':filtorder[f]*100+order, 'caption':
                         'Minimum airmass in %s.' %(propCaption)}
            histMerge={'color':tcolor, 'label':'%s %s' %(f, tlabel),
                       'binsize':0.03, 'legendloc':'upper right'}
            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, runName=runName, metadata=metadata,
                                            summaryMetrics=summaryStats)
            mergedHistDict['minAirmass'].addBundle(bundle,plotDict=histMerge)
            bundleList.append(bundle)

            metric= metrics.FracAboveMetric(col='airmass', cutoff=airmass_limit)
            plotDict={'xMin':0, 'xMax':1, 'color':tcolor}
            summaryStats=allStats
            displayDict={'group':seeinggroup, 'subgroup':'Low airmass fraction',
                         'order':filtorder[f]*100+order, 'caption':
                         'Fraction of total images with airmass higher than %.2f, in %s'
                         %(airmass_limit, propCaption)}
            histMerge={'color':tcolor, 'label':'%s %s' %(f, tlabel), 'binsize':0.05, 'legendloc':'upper right'}

            bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                                displayDict=displayDict, runName=runName, metadata=metadata,
                                                summaryMetrics=summaryStats)
            mergedHistDict['fracAboveAirmass'].addBundle(bundle,plotDict=histMerge)
            bundleList.append(bundle)

# SNe metrics from UK workshop.
    peaks = {'uPeak':25.9, 'gPeak':23.6, 'rPeak':22.6, 'iPeak':22.7, 'zPeak':22.7,'yPeak':22.8}
    peakTime = 15.
    transDuration = peakTime+30. # Days
    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          metricName='SNDetection',**peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected at any point in their light curve in any filter'
    displayDict={'group':sngroup,  'subgroup':'Detected', 'caption':caption}
    sqlconstraint = ''
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.0,
                                          transDuration=transDuration, peakTime=peakTime,
                                          surveyDuration=runLength,
                                          nPrePeak=1, metricName='SNAlert', **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are detected pre-peak in any filter'
    displayDict={'group':sngroup,  'subgroup':'Detected on the rise', 'caption':caption}
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)

    metric = metrics.TransientMetric(riseSlope= -2./peakTime, declineSlope=1.4/30.,
                                     transDuration=transDuration, peakTime=peakTime,
                                     surveyDuration=runLength, metricName='SNLots',
                                     nFilters=3, nPrePeak=3, nPerLC=2, **peaks)
    caption = 'Fraction of z=0.5 type Ia SN that are observed 6 times, 3 pre-peak, 3 post-peak, with observations in 3 filters'
    displayDict={'group':sngroup,  'subgroup':'Well observed', 'caption':caption}
    sqlconstraint = 'filter="r" or filter="g" or filter="i" or filter="z" '
    plotDict={}
    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)


    # Full range of dates:
    metric = metrics.FullRangeMetric(col='expMJD')
    plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
    caption='Time span of survey.'
    sqlconstraint = ''
    plotDict={}
    displayDict={'group':rangeGroup, 'caption':caption}

    bundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint, plotDict=plotDict,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)
    for filt in filters:
        for propid in propids:
            displayDict={'group':rangeGroup, 'subgroup':propids[propid] ,'caption':caption}
            md = '%s, %s' % (filt, propids[propid])
            sql = 'filter="%s" and propID=%i' % (filt,propid)
            bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                                metadata=md, plotFuncs=plotFuncs,
                                                displayDict=displayDict, runName=runName)
            bundleList.append(bundle)



    # Alt az plots
    slicer = slicers.HealpixSlicer(nside=64, latCol='zenithDistance', lonCol='azimuth', useCache=False)
    metric = metrics.CountMetric('expMJD', metricName='Nvisits as function of Alt/Az')
    plotDict = {}
    plotFuncs = [plots.LambertSkyMap()]
    displayDict = {'group':altAzGroup, 'caption':'Alt Az pointing distribution'}
    for filt in filters:
        for propid in propids:
            displayDict = {'group':altAzGroup, 'subgroup':propids[propid], 'caption':'Alt Az pointing distribution'}
            md = '%s, %s' % (filt, propids[propid])
            sql = 'filter="%s" and propID=%i' % (filt,propid)
            bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                                plotFuncs=plotFuncs, metadata=md,
                                                displayDict=displayDict, runName=runName)
            bundleList.append(bundle)

    sql = ''
    md='all observations'
    displayDict = {'group':altAzGroup, 'subgroup':'All Observations',
                   'caption':'Alt Az pointing distribution'}
    bundle = metricBundles.MetricBundle(metric,slicer,sql, plotDict=plotDict,
                                        plotFuncs=plotFuncs, metadata=md,
                                        displayDict=displayDict, runName=runName)
    bundleList.append(bundle)


    # Median inter-night gap (each and all filters)
    slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
    metric = metrics.InterNightGapsMetric(metricName='Median Inter-Night Gap')
    displayDict = {'group':intergroup, 'subgroup': 'Median Gap','caption':'Median gap between days'}
    sqls = ['filter = "%s"' % f for f in filters]
    sqls.append('')
    for sql in sqls:
        bundle = metricBundles.MetricBundle(metric, slicer, sql, displayDict=displayDict, runName=runName)
        bundleList.append(bundle)

    # Max inter-night gap in r and all bands
    dslicer = slicers.HealpixSlicer(nside=nside, lonCol='ditheredRA', latCol='ditheredDec')
    metric = metrics.InterNightGapsMetric(metricName='Max Inter-Night Gap', reduceFunc=np.max)
    displayDict = {'group':intergroup, 'subgroup':'Max Gap', 'caption':'Max gap between nights'}
    plotDict = {'percentileClip':95.}
    for sql in sqls:
        bundle = metricBundles.MetricBundle(metric, dslicer, sql, displayDict=displayDict,
                                            plotDict=plotDict, runName=runName)
        bundleList.append(bundle)


    # largest phase gap for periods
    periods = [0.1,1.0,10.,100.]
    sqls = {'u':'filter = "u"', 'r':'filter="r"',
            'g,r,i,z':'filter="g" or filter="r" or filter="i" or filter="z"',
            'all':''}

    for sql in sqls.keys():
        for period in periods:
            displayDict = {'group':phaseGroup,
                           'subgroup':'period=%.2f days, filter=%s' % (period,sql),
                           'caption':'Maximum phase gaps'}
            metric = metrics.PhaseGapMetric(nPeriods=1, periodMin=period, periodMax=period,
                                            metricName='PhaseGap, %.1f'%period)
            bundle = metricBundles.MetricBundle(metric, slicer, sqls[sql],
                                                displayDict=displayDict, runName=runName)
            bundleList.append(bundle)




    # NEO XY plots
    slicer = slicers.UniSlicer()
    metric = metrics.PassMetric(metricName='NEODistances')
    stacker = stackers.NEODistStacker()
    stacker2 = stackers.EclipticStacker()
    for f in filters:
        plotFunc = plots.NeoDistancePlotter(eclipMax=10., eclipMin=-10.)
        displayDict = {'group': NEOGroup, 'subgroup':'xy',
                       'caption':'Observations within 10 degrees of the ecliptic. Distance an H=22 NEO would be detected'}
        plotDict={}
        sqlconstraint = 'filter = "%s"' %(f)
        bundle = metricBundles.MetricBundle(metric, slicer,
                                            sqlconstraint, displayDict=displayDict,
                                            stackerList=[stacker,stacker2],
                                            plotDict=plotDict,
                                            plotFuncs=[plotFunc])
        noSaveBundleList.append(bundle)


    # Solar elongation
    sqls = ['filter = "%s"' % f for f in filters]
    sqls.append('')
    for sql in sqls:
        plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
        displayDict = {'group': NEOGroup, 'subgroup':'Solar Elongation',
                           'caption':'Median solar elongation in degrees'}
        metric = metrics.MedianMetric('solarElong')
        slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
        bundle = metricBundles.MetricBundle(metric, slicer,sql, displayDict=displayDict, plotFuncs=plotFuncs)
        bundleList.append(bundle)

        plotFuncs = [plots.HealpixSkyMap(), plots.HealpixHistogram()]
        displayDict = {'group': NEOGroup, 'subgroup':'Solar Elongation',
                           'caption':'Minimum solar elongation in degrees'}
        metric = metrics.MinMetric('solarElong')
        slicer = slicers.HealpixSlicer(nside=nside, lonCol=lonCol, latCol=latCol)
        bundle = metricBundles.MetricBundle(metric, slicer,sql, displayDict=displayDict, plotFuncs=plotFuncs)
        bundleList.append(bundle)


    return metricBundles.makeBundlesDictFromList(bundleList), mergedHistDict, metricBundles.makeBundlesDictFromList(noSaveBundleList)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the science performance metrics')
    parser.add_argument('dbFile', type=str, default=None,help="full file path to the opsim sqlite file")

    parser.add_argument("--outDir",type=str, default='./Out', help='Output directory for MAF outputs. Default "Out"')
    parser.add_argument("--nside", type=int, default=64,
                        help="Resolution to run Healpix grid at (must be 2^x). Default 64.")
    parser.add_argument("--lonCol", type=str, default='fieldRA',
                        help="Column to use for RA values (can be a stacker dither column). Default=fieldRA.")
    parser.add_argument("--latCol", type=str, default='fieldDec',
                        help="Column to use for Dec values (can be a stacker dither column). Default=fieldDec.")
    parser.add_argument('--seeingCol', type=str, default='FWHMgeom',
                        help="Column to use for seeing values in order to evaluate astrometric uncertainties. Probably should be FWHMgeom or finSeeing.")
    parser.add_argument('--benchmark', type=str, default='design',
                        help="Can be 'design' or 'requested'")
    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values from disk and re-plot them.")
    parser.add_argument('--skipNoSave', dest='runNoSave', action='store_false', default=True,
                        help="Skip the metrics that do not get saved as npz files.")

    parser.set_defaults()
    args, extras = parser.parse_known_args()

    # Build metric bundles.

    bundleDict, mergedHistDict, noSaveBundleDict = makeBundleList(args.dbFile, nside=args.nside,
                                                                  lonCol=args.lonCol, latCol=args.latCol,
                                                                  benchmark=args.benchmark,
                                                                  seeingCol=args.seeingCol)

    # Set up / connect to resultsDb.
    resultsDb = db.ResultsDb(outDir=args.outDir)
    # Connect to opsimdb.
    opsdb = utils.connectOpsimDb(args.dbFile)

    if args.runNoSave:
        group = metricBundles.MetricBundleGroup(noSaveBundleDict, opsdb, saveEarly=False,
                                                outDir=args.outDir, resultsDb=resultsDb)
        group.runAll(clearMemory=True, plotNow=True)
        del group, noSaveBundleDict

    # Set up metricBundleGroup.
    group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
                                            outDir=args.outDir, resultsDb=resultsDb)
    # Read or run to get metric values.
    if args.plotOnly:
        group.readAll()
    else:
        group.runAll()
    # Make plots.
    group.plotAll()
    # Make merged plots.
    for key in mergedHistDict:
        if len(mergedHistDict[key].bundleList) > 0:
            mergedHistDict[key].plot(outDir=args.outDir, resultsDb=resultsDb, closeFigs=True)
        else:
            warning.warn('Empty bundleList for %s, skipping merged histogram' % key)
    # Get config info and write to disk.
    utils.writeConfigs(opsdb, args.outDir)

    print "Finished sciencePerformance metric calculations."
