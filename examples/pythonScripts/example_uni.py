## EXAMPLE
# example test script for unislicer metrics.


import argparse
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.db as db

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


def getMetrics():
    t = time.time()
    # Set up metrics.
    metricList = []
    # Simple metrics:
    metricList.append(metrics.MeanMetric('finSeeing'))
    metricList.append(metrics.RmsMetric('finSeeing'))
    metricList.append(metrics.MedianMetric('airmass'))
    metricList.append(metrics.RmsMetric('airmass'))
    metricList.append(metrics.MeanMetric('fiveSigmaDepth'))
    metricList.append(metrics.RmsMetric('fiveSigmaDepth'))
    metricList.append(metrics.MeanMetric('filtSkyBrightness'))
    metricList.append(metrics.CountMetric('expMJD'))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getSlicer(simdata):
    t = time.time()
    bb = slicers.UniSlicer()
    bb.setupSlicer(simdata)

    dt, t = dtime(t)
    print 'Set up slicer %f s' %(dt)
    return bb

def goSlice(opsimrun, metadata, simdata, bb, metricList):
    t = time.time()
    gm = sliceMetrics.RunSliceMetric()
    gm.setSlicer(bb)

    gm.setMetrics(metricList)
    gm.runSlices(simdata, simDataName=opsimrun, metadata=metadata)
    dt, t = dtime(t)
    print 'Ran bins of %d points with %d metrics using sliceMetric %f s' %(len(bb), len(metricList), dt)

    gm.reduceAll()

    dt, t = dtime(t)
    print 'Ran reduce functions %f s' %(dt)

    return gm


def write(gm):
    t= time.time()
    gm.writeAll()
    dt, t = dtime(t)
    print 'Wrote outputs %f s' %(dt)

def printSummary(gm, metricList):
    t = time.time()
    for m in metricList:
       iid = gm.metricObjIid(m)[0]
       value = gm.computeSummaryStatistics(iid, metrics.MeanMetric(''))
       print 'Summary for', m.name, ':', value
    dt, t = dtime(t)
    print 'Computed summaries %f s' %(dt)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("simDataTable", type=str, help="Filename (with path) of sqlite database")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    args = parser.parse_args()

    # Get db connection info.
    dbAddress = 'sqlite:///' + args.simDataTable
    oo = db.OpsimDatabase(dbAddress)

    opsimrun = oo.fetchOpsimRunName()

    sqlconstraint = args.sqlConstraint

    # Set up metrics.
    metricList = getMetrics()

    # Find columns that are required.
    colnames = list(metricList[0].colRegistry.colSet)

    # Get opsim simulation data
    simdata = oo.fetchMetricData(colnames, sqlconstraint)

    # And set up slicer.
    bb = getSlicer(simdata)

    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"', '')
    gm = goSlice(opsimrun, metadata, simdata, bb, metricList)

    # Generate some summary statistics and plots.
    printSummary(gm, metricList)

    # Unlike other examples, don't generate any plots (these are single number results).

    # Write the data to file.
    write(gm)

