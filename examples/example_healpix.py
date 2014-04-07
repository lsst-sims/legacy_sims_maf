## EXAMPLE
# example test script for healpix metrics. 
# Note that this is not expected to function as the driver! It just has some command line options.

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics
import glob

from lsst.sims.catalogs.generation.db.utils import make_engine
from lsst.sims.operations.maf.utils import getData

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


def getMetrics(seeingcol, docomplex=True):
    t = time.time()
    # Set up metrics.
    metricList = []
    # Simple metrics: 
    metricList.append(metrics.MeanMetric(seeingcol))
    metricList.append(metrics.MedianMetric('airmass'))
    metricList.append(metrics.MinMetric('airmass'))
    metricList.append(metrics.MeanMetric('5sigma_modified'))
    metricList.append(metrics.MeanMetric('skybrightness_modified'))
    metricList.append(metrics.Coaddm5Metric('5sigma_modified'))
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                          plotParams={'ylog':False, 'title':'Number of visits',
                                                      'plotMin':0, 'plotMax':300,
                                                      'cbarFormat': '%d'}))
    if docomplex:
        # More complex metrics.    
        dtmin = 1./60./24.
        dtmax = 360./60./24.
        metricList.append(metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax,
                                                   plotParams={'ylog':False, 'plotMin':0, 'plotMax':20}))

    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getBinner(simdata, racol, deccol, nside=128, leafsize=100):
    t = time.time()
    bb = binners.HealpixBinner(nside=nside, spatialkey1=racol, spatialkey2=deccol)    
    bb.setupBinner(simdata, leafsize=leafsize)
    dt, t = dtime(t)
    print 'Set up binner and built kdtree %f s' %(dt)
    return bb


def goBin(opsimrun, metadata, simdata, bb, metricList):
    t = time.time()
    gm = binMetrics.BaseBinMetric()
    gm.setBinner(bb)
    
    dt, t = dtime(t)
    print 'Set up gridMetric %f s' %(dt)

    gm.setMetrics(metricList)
    gm.runBins(simdata, simDataName=opsimrun, metadata=metadata)
    dt, t = dtime(t)
    print 'Ran bins of %d points with %d metrics using binMetric %f s' %(len(bb), len(metricList), dt)
                    
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
            mean = gm.computeSummaryStatistics(m.name, metrics.MeanMetric)
            rms = gm.computeSummaryStatistics(m.name, metrics.RmsMetric)
            print 'Summary for', m.name, ':\t Mean', mean, '\t rms', rms
        except Exception as e:
            # Probably have a metric data value which does not 'work' for the mean metric.
            print ' Cannot compute mean or rms for metric values', m.name
    dt, t = dtime(t)
    print 'Computed summaries %f s' %(dt)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("simDataTable", type=str, help="Name of opsim visit table in database")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--nside", type=int, default=128,
                        help="NSIDE parameter for healpix grid resolution. Default 128.")
    parser.add_argument("--dither", dest='dither', action='store_true',
                        help="Use hexdither RA/Dec values.")
    parser.set_defaults(dither=False)
    parser.add_argument("--connectionName", type=str, default='SQLITE_OPSIM', 
                       help="Key for the connection string to use in your dbLogin file -- "\
                            "Default is SQLITE_OPSIM")
    args = parser.parse_args()
    
    # Get db connection info.
    dbAddress = getData.getDbAddress(connectionName = args.connectionName)
    
    dbTable = args.simDataTable
    opsimrun = args.simDataTable.replace('output_', '')

    sqlconstraint = args.sqlConstraint
    
    # Bit of a kludge to set seeing column name. 
    table = db.Table(dbTable, 'obsHistID', dbAddress)
    try:
        table.query_columns_RecArray(colnames=['seeing',], numLimit=1)
        seeingcol = 'seeing'
    except ValueError:
        try:
            table.query_columns_RecArray(colnames=['finSeeing',], numLimit=1)
            seeingcol = 'finSeeing'
        except ValueError:
            raise ValueError('Cannot find appropriate column name for seeing.')
    print 'Using %s for seeing column name.' %(seeingcol)
    
    # Set up metrics. 
    metricList = getMetrics(seeingcol, docomplex=True)

    # Find columns that are required.
    colnames = list(metricList[0].classRegistry.uniqueCols())
    fieldcols = ['fieldRA', 'fieldDec', 'hexdithra', 'hexdithdec']
    colnames = colnames + fieldcols
    colnames = list(set(colnames))
    
    # Get opsim simulation data
    simdata = getData.fetchSimData(dbTable, dbAddress, sqlconstraint, colnames)
    
    # And set up binner.
    if args.dither:
        racol = 'hexdithra'
        deccol = 'hexdithdec'
    else:
        racol = 'fieldRA'
        deccol = 'fieldDec'
    # Check which kdtree is available and set leafsize
    from scipy.spatial import cKDTree as kdtree
    if not hasattr(kdtree,'query_ball_point'):
       leafsize=50000
    else:
       leafsize=100
    bb = getBinner(simdata, racol, deccol, args.nside, leafsize=leafsize)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','')
    if args.dither:
        metadata = metadata + ' hexdither'
    gm = goBin(opsimrun, metadata, simdata, bb, metricList)

    # Generate some summary statistics and plots.
    printSummary(gm, metricList)
    # Generate (and save) plots.
    plot(gm)

    # Write the data to file.
    write(gm)
    
