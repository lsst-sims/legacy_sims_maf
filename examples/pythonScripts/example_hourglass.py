## EXAMPLE

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.binners as binners
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binMetrics as binMetrics
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

def getBinner(simdata, metricList, nbins=100):
    t = time.time()
    binnerList = []
    for m in metricList:
        bb = binners.HourglassBinner()
        bb.setupBinner(simdata)
        binnerList.append(bb)
    dt, t = dtime(t)
    print 'Set up binners %f s' %(dt)
    return binnerList


def goBinPlotWrite(opsimrun, metadata, simdata, binnerList, metricList):
    t = time.time()
    for bb, mm in zip(binnerList, metricList):
        gm = binMetrics.BaseBinMetric()
        gm.setBinner(bb)
        gm.setMetrics(mm)
        gm.runBins(simdata, simDataName=opsimrun, metadata=metadata)
        gm.plotAll(savefig=True, closefig=True)
        gm.writeAll()
        dt, t = dtime(t)
        print 'Ran bins of %d points with %d metrics using binMetric %f s' %(len(bb), len([mm,]), dt)


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
    colnames = list(metricList[0].classRegistry.uniqueCols())

    # Get opsim simulation data
    simdata = oo.fetchMetricData(colnames, sqlconstraint)
    
    # And set up binner.
    binnerList = getBinner(simdata, metricList)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'')
    gm = goBinPlotWrite(opsimrun, metadata, simdata, binnerList, metricList)
