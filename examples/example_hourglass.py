## EXAMPLE
# example test script for oneD metrics. 
# Note that this is not expected to function as the driver! It just has some command line options.

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.binners as binners
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binMetrics as binMetrics
import glob

from lsst.sims.catalogs.generation.db.utils import make_engine
from lsst.sims.maf.utils import getData

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

def getDbAddress():
    # Get the database connection information from the dbLogin file in the user's home directory.
    home_path = os.getenv("HOME")
    f=open("%s/dbLogin"%(home_path),"r")
    authDictionary = {}
    for l in f:
        els = l.rstrip().split()
        authDictionary[els[0]] = els[1]
    return authDictionary

def getMetrics(seeingcol):
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
    parser.add_argument("simDataTable", type=str, help="Name of opsim visit table in database")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--connectionName", type=str, default='SQLITE_OPSIM', 
                       help="Key for the connection string to use in your dbLogin file -- "\
                            "Default is SQLITE_OPSIM")
    args = parser.parse_args()
    
    # Get db connection info.
    authDictionary = getDbAddress()
    dbAddress = authDictionary[args.connectionName]
    
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
    metricList = getMetrics(seeingcol)

    # Find columns that are required.
    colnames = list(metricList[0].classRegistry.uniqueCols())

    # Get opsim simulation data
    simdata = getData.fetchSimData(dbTable, dbAddress, sqlconstraint, colnames)
    
    # And set up binner.
    binnerList = getBinner(simdata, metricList)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'')
    gm = goBinPlotWrite(opsimrun, metadata, simdata, binnerList, metricList)
