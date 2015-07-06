import numpy as np
import os, sys, argparse
# Set matplotlib backend (to create plots where DISPLAY is not set).
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles


def makeBundleList(dbFile, nside=128, benchmark='design',):
    """
    make a list of metricBundle objects to look at the scientific performance
    of an opsim run.
    """

    # List to hold everything we're going to make
    bundleList = []

    # Connect to the databse
    opsimdb = utils.connectOpsimDb(dbFile)

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

    # Set up a list of common summary stats
    commonSummary = [metrics.MeanMetric(), metrics.RobustRmsMetric(), metrics.MedianMetric(),
                     metrics.PercentileMetric(metricName='25th%ile', percentile=25),
                     metrics.PercentileMetric(metricName='75th%ile', percentile=75),
                     metrics.MinMetric(), metrics.MaxMetric()]


    # Set up some 'group' labels
    reqgroup = 'A: Required SRD metrics'
    depthgroup = 'B: Depth per filter'
    uniformitygroup = 'C: Uniformity'
    seeinggroup = 'D: Seeing distribution'

    histNum = 0

    # Calculate the fO metrics for all proposals and WFD only.
    order = 0
    for prop in ('All prop', 'WFD only'):
        if prop == 'All prop':
            metadata = 'All Visits' + slicermetadata
            sqlconstraint = ['']
        if prop == 'WFD only':
            metadata = 'WFD only' + slicermetadata
            sqlconstraint = ['%s' %(wfdWhere)]
        # Configure the count metric which is what is used for f0 slicer.
        m1 = metrics.CountMetric(col='expMJD', metricName='fO')
        plotDict={'units':'Number of Visits','Asky':benchmarkVals['Area'],
                  'Nvisit':benchmarkVals['nvisitsTotal'],
                  'xMin':0, 'xMax':1500})
        summaryMetrics={'fOArea 1':{'nside':nside, 'norm':False, 'metricName':'fOArea: Nvisits (#)',
                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                        'fOArea 2':{'nside':nside, 'norm':True, 'metricName':'fOArea: Nvisits/benchmark',
                                    'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                        'fONv 1':{'nside':nside, 'norm':False, 'metricName':'fONv: Area (sqdeg)',
                                  'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']},
                        'fONv 2':{'nside':nside, 'norm':True, 'metricName':'fONv: Area/benchmark',
                                  'Asky':benchmarkVals['Area'],'Nvisit':benchmarkVals['nvisitsTotal']}}
        displayDict={'group':reqgroup, 'subgroup':'F0', 'displayOrder':order, 'caption':
                     'The FO metric evaluates the overall efficiency of observing. fOArea: Nvisits = %.1f sq degrees receive at least this many visits out of %d. fONv: Area = this many square degrees out of %.1f receive at least %d visits.'
                                        %(benchmarkVals['Area'], benchmarkVals['nvisitsTotal'],
                                          benchmarkVals['Area'], benchmarkVals['nvisitsTotal'])}
        order += 1
        slicer = slicers.fOSlicer(metadata=metadata, metadataVerbatim=True)

        bundle = metricBundles.MetricBundle(m1, slicer, sqlconstraint, plotDict=plotDict,
                                            displayDict=displayDict, summaryMetrics=summaryMetrics)
        bundleList.append(bundle)


    # For each bundle, we need to set a metric, a slicer, a sql constraint, and optionally a plotDict.


    return bundleList


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Python script to run MAF with the science performance metrics')
    parser.add_argument("--outDir",type=str, default='./Out', help='Output directory for MAF outputs.')
    parser.add_argument("--nside", type=int, default=128,
                        help="Resolution to run Healpix grid at (must be 2^x)")

    parser.add_argument('--dbFile', type=str, default=None,help="full file path to the opsim sqlite file")

    parser.add_arguemnt('--benchmark', type=str, default='design',
                        help="Can be 'design' or 'requested'")

    parser.set_defaults()
    args, extras = parser.parse_known_args()


    bundleList = makeBundleList(args.dbFile,nside=args.nside, benchmark=args.benchmark)


    resultsDb = db.ResultsDb(outDir=args.outDir)
