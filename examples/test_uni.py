## EXAMPLE
# example test script for unibinner metrics. 
# Note that this is not expected to function as the driver! It just has some command line options.

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics

from lsst.sims.catalogs.generation.db.utils import make_engine
from lsst.sims.operations.maf.utils import getData

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
    metricList.append(metrics.MeanMetric(seeingcol))
    metricList.append(metrics.RmsMetric(seeingcol))
    metricList.append(metrics.MedianMetric('airmass'))
    metricList.append(metrics.RmsMetric('airmass'))
    metricList.append(metrics.MeanMetric('5sigma_modified'))
    metricList.append(metrics.RmsMetric('5sigma_modified'))
    metricList.append(metrics.MeanMetric('skybrightness_modified'))
    metricList.append(metrics.CountMetric('expMJD'))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList

def getBinner(simdata):
    t = time.time()
    bb = binners.UniBinner()
    bb.setupBinner(simdata)
    
    dt, t = dtime(t)
    print 'Set up binner %f s' %(dt)
    return bb

def goBin(dbTable, metadata, simdata, bb, metricList):
    t = time.time()
    gm = binMetrics.BaseBinMetric()
    gm.setBinner(bb)

    gm.setMetrics(metricList)
    gm.runBins(simdata, simDataName=dbTable, metadata = metadata)
    dt, t = dtime(t)
    print 'Ran bins of %d points with %d metrics using binMetric %f s' %(len(bb), len(metricList), dt)
                    
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
        try:
            value = gm.computeSummaryStatistics(m.name, None)
            print 'Summary for', m.name, ':', value
        except ValueError:
            pass
    dt, t = dtime(t)
    print 'Computed summaries %f s' %(dt)

    
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
    opsimrun = args.simDataTable.lstrip('output_')

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
    bb = getBinner(simdata)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'')
    gm = goBin(dbTable, metadata, simdata, bb, metricList)

    # Generate some summary statistics and plots.
    printSummary(gm, metricList)

    # No plots for unibinner (these are single number results).
    
    # Write the data to file.
    write(gm)
    
