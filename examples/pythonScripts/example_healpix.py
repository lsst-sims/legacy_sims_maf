## EXAMPLE
# example of interacting directly with the python classes, for a healpix slicer.
# to run:
#python example_healpix.py ../../tests/opsimblitz1_1131_sqlite.db

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import glob


import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


def getMetrics(docomplex=False):
    t = time.time()
    # Set up metrics.
    metricList = []
    # Simple metrics: 
    metricList.append(metrics.MeanMetric('finSeeing'))
    metricList.append(metrics.MedianMetric('airmass'))
    metricList.append(metrics.MinMetric('airmass'))
    metricList.append(metrics.MeanMetric('fivesigma_modified'))
    metricList.append(metrics.MeanMetric('skybrightness_modified'))
    metricList.append(metrics.Coaddm5Metric('fivesigma_modified'))
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                          plotParams={'ylog':False, 'title':'Number of visits',
                                                      'plotMin':0, 'plotMax':300,
                                                      'cbarFormat': '%d'}))
    if docomplex:
        # More complex metrics.    
        dtmin = 1./60./24.
        dtmax = 360./60./24.
        metricList.append(metrics.VisitGroupsMetric(deltaTmin=dtmin, deltaTmax=dtmax,
                                                    plotParams={'ylog':False, 'plotMin':0, 'plotMax':20}))
        
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getSlicer(simdata, racol, deccol, nside=128):
    t = time.time()
    bb = slicers.HealpixSlicer(nside=nside, spatialkey1=racol, spatialkey2=deccol)    
    bb.setupSlicer(simdata)
    dt, t = dtime(t)
    print 'Set up slicer and built kdtree %f s' %(dt)
    return bb


def goBin(opsimrun, metadata, simdata, bb, metricList):
    t = time.time()
    gm = sliceMetrics.BaseSliceMetric()
    gm.setSlicer(bb)
    
    dt, t = dtime(t)
    print 'Set up gridMetric %f s' %(dt)

    gm.setMetrics(metricList)
    gm.runSlices(simdata, simDataName=opsimrun, metadata=metadata)
    dt, t = dtime(t)
    print 'Ran bins of %d points with %d metrics using sliceMetric %f s' %(len(bb), len(metricList), dt)
                    
    gm.reduceAll()
    
    dt, t = dtime(t)
    print 'Ran reduce functions %f s' %(dt)

    return gm

def plot(gm):
    t = time.time()
    gm.plotAll(savefig=True, closefig=True, verbose=True)
    
    dt, t = dtime(t)
    print 'Made plots %f s' %(dt)

def write(gm):
    t= time.time()
    gm.writeAll()
    dt, t = dtime(t)
    print 'Wrote outputs %f s' %(dt)

def printSummary(gm, metricList):
    t = time.time()
    for m in metricList:
        try:
            mean = gm.computeSummaryStatistics(m.name, metrics.MeanMetric(''))
            rms = gm.computeSummaryStatistics(m.name, metrics.RmsMetric(''))
            print 'Summary for', m.name, ':\t Mean', mean, '\t rms', rms
        except Exception as e:
            # Probably have a metric data value which does not 'work' for the mean metric.
            print ' Cannot compute mean or rms for metric values', m.name
    dt, t = dtime(t)
    print 'Computed summaries %f s' %(dt)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Filename for opsim sqlite db file")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--nside", type=int, default=128,
                        help="NSIDE parameter for healpix grid resolution. Default 128.")
    parser.add_argument("--dither", dest='dither', action='store_true',
                        help="Use hexdither RA/Dec values.")
    parser.set_defaults(dither=False)
    args = parser.parse_args()
    
    # Get db connection info.
    dbAddress = 'sqlite:///' + args.opsimDb
    oo = db.OpsimDatabase(dbAddress)

    opsimrun = oo.fetchOpsimRunName()

    sqlconstraint = args.sqlConstraint
    
    
    # Set up metrics. 
    metricList = getMetrics(docomplex=False)

    # Find columns that are required.
    colnames = list(metricList[0].classRegistry.uniqueCols())
    fieldcols = ['fieldRA', 'fieldDec', 'hexdithra', 'hexdithdec']
    colnames = colnames + fieldcols
    colnames = list(set(colnames))
    
    # Get opsim simulation data
    simdata = oo.fetchMetricData(colnames, sqlconstraint)
    
    # And set up slicer.
    if args.dither:
        racol = 'hexdithra'
        deccol = 'hexdithdec'
    else:
        racol = 'fieldRA'
        deccol = 'fieldDec'

    bb = getSlicer(simdata, racol, deccol, args.nside)
    
    # Okay, go calculate the metrics.
    comment = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')
    if args.dither:
        metadata = metadata + ' hexdither'
    gm = goBin(opsimrun, comment, simdata, bb, metricList)

    # Generate some summary statistics and plots.
    printSummary(gm, metricList)
    # Generate (and save) plots.
    plot(gm)

    # Write the data to file.
    write(gm)
    
