## EXAMPLE


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


def getMetrics():
    t = time.time()
    # Set up metrics.
    metricList = []
    # Simple metrics: 
    metricList.append(metrics.CountMetric('finSeeing'))
    metricList.append(metrics.CountMetric('airmass'))
    metricList.append(metrics.CountMetric('fivesigma_modified'))
    metricList.append(metrics.CountMetric('skybrightness_modified'))
    metricList.append(metrics.CountMetric('expMJD'))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getSlicer(simdata, metricList, bins=100):
    t = time.time()
    slicerList = []
    for m in metricList:
        bb = slicers.OneDSlicer(sliceColName=m.colname, bins=bins)
        bb.setupSlicer(simdata)
        slicerList.append(bb)
    dt, t = dtime(t)
    print 'Set up slicers %f s' %(dt)
    return slicerList


def goSlicePlotWrite(opsimrun, metadata, simdata, slicerList, metricList):
    t = time.time()
    for bb, mm in zip(slicerList, metricList):
        gm = sliceMetrics.BaseSliceMetric()
        gm.setSlicer(bb)
        gm.setMetrics(mm)
        gm.runSlices(simdata, simDataName=opsimrun, metadata=metadata)
        mean = gm.computeSummaryStatistics(mm.name, metrics.SumMetric(''))
        print 'SummaryNumber (sum) for', mm.name, ':', mean
        gm.plotAll(savefig=True, closefig=True)
        gm.writeAll()
        dt, t = dtime(t)
        print 'Ran bins of %d points with %d metrics using sliceMetric %f s' %(len(bb), len([mm,]), dt)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Filename of sqlite database")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    args = parser.parse_args()
    
    # Get db connection info.
    dbAddress = 'sqlite:///' + args.opsimDb
    oo = db.OpsimDatabase(dbAddress)

    opsimrun = oo.fetchOpsimRunName()

    sqlconstraint = args.sqlConstraint
    
     
    # Set up metrics. 
    metricList = getMetrics()

    # Find columns that are required.
    colnames = list(metricList[0].classRegistry.uniqueCols())

    # Get opsim simulation data
    simdata = oo.fetchMetricData(colnames, sqlconstraint)
    
    # And set up slicer.
    slicerList = getSlicer(simdata, metricList)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','')
    gm = goSlicePlotWrite(opsimrun, metadata, simdata, slicerList, metricList)
