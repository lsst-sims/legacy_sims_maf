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
    metricList.append(metrics.HourglassMetric())
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getSlicer(simdata, metricList, nbins=100):
    t = time.time()
    slicerList = []
    for m in metricList:
        bb = slicers.HourglassSlicer()
        bb.setupSlicer(simdata)
        slicerList.append(bb)
    dt, t = dtime(t)
    print 'Set up slicers %f s' %(dt)
    return slicerList


def goSlicePlotWrite(opsimrun, metadata, simdata, slicerList, metricList):
    t = time.time()
    for bb, mm in zip(slicerList, metricList):
        gm = sliceMetrics.RunSliceMetric()
        gm.setSlicer(bb)
        gm.setMetrics(mm)
        gm.runSlices(simdata, simDataName=opsimrun, metadata=metadata)
        gm.plotAll(savefig=True, closefig=True)
        gm.writeAll()
        dt, t = dtime(t)
        print 'Ran bins of %d points with %d metrics using sliceMetric %f s' %(len(bb), len([mm,]), dt)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("simDataTable", type=str, help="Filename of opsim database")
    parser.add_argument("--sqlConstraint", type=str, default="", help="SQL constraint")
    args = parser.parse_args()
    
    # Get opsim info / db connection.
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
    slicerList = getSlicer(simdata, metricList)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'')
    gm = goSlicePlotWrite(opsimrun, metadata, simdata, slicerList, metricList)
