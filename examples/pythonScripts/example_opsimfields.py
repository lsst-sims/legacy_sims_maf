# Example/test script for using the opsimField binner. 
# This is not intended to replace the driver, but can serve as an example of using MAF
#  directly through python. 

import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.binners as binners
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binMetrics as binMetrics
import lsst.sims.maf.utils as utils

from lsst.sims.catalogs.generation.db.utils import make_engine
from lsst.sims.maf.utils import getData

import glob

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
                                          plotParams={'ylog':False, '_units':'Number of visits',
                                                      'plotMin':0, 'plotMax':300}))
    if docomplex:
        # More complex metrics.    
        dtmin = 1./60./24.
        dtmax = 360./60./24.
        metricList.append(metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList
    
def getBinner(simData, fieldDataInfo):
    # Setting up the binner will be slightly different for each binner.
    t = time.time()    
    bb = binners.OpsimFieldBinner(simDataFieldIdColName='fieldID', fieldIdColName='fieldID',
                                  fieldRaColName='fieldRA', fieldDecColName='fieldDec')
    if not(fieldDataInfo['useFieldTable']): # use simData / output table for fieldData
        fieldID, idx = np.unique(simData['fieldID'], return_index=True)
        ra = simData['fieldRA'][idx]
        dec = simData['fieldDec'][idx]
        fieldData = np.core.records.fromarrays([fieldID, ra, dec],
                                               names=['fieldID', 'fieldRA', 'fieldDec'])
    if (fieldDataInfo['useFieldTable']): # use field table to get Field Data
        fieldData = utils.getData.fetchFieldsFromFieldTable(fieldDataInfo['dbAddress'],
                                                            fieldTable=fieldDataInfo['fieldTable'],
                                                            proposalFieldTable=fieldDataInfo['proposalTable'],
                                                            proposalID=fieldDataInfo['proposalID'],
                                                            sessionID=fieldDataInfo['sessionID'])
    # SetUp binner.
    bb.setupBinner(simData, fieldData)
    dt, t = dtime(t)
    print 'Set up binner %f s' %(dt)
    return bb

def goBin(opsimrun, metadata, simdata, bb, metricList):
    t = time.time()
    gm = binMetrics.BaseBinMetric()
    gm.setBinner(bb)
    
    dt, t = dtime(t)
    print 'Set up gridMetric %f s' %(dt)

    gm.setMetrics(metricList)
    gm.runBins(simdata, simDataName=opsimrun, metadata = metadata)
    dt, t = dtime(t)
    print 'Ran bins of %d points with %d metrics using binMetric %f s' %(len(bb), len(metricList), dt)
                    
    gm.reduceAll()
    
    dt, t = dtime(t)
    print 'Ran reduce functions %f s' %(dt)

    return gm

def plot(gm):
    t = time.time()
    gm.plotAll(savefig=True, closefig=True)
    
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
    parser.add_argument("simDataTable", type=str, help="Name of opsim visit table in database")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--useFieldTable", type=bool, default=False, 
                        help="Use opsim Field table to get field pointing info (default False)")
    parser.add_argument("--fieldDataTable", type=str, default='Field',
                        help="Name of the field visit table in db --"\
                        "Default is Field.")
    parser.add_argument("--propID", type=int, default=-666, 
                        help="Proposal ID number if using propID as a constraint.")
    parser.add_argument("--proposalFieldTable", type=str, default='tProposal_Field', 
                        help="Name of the Proposal_Field table, if adding propID as constraint.")
    parser.add_argument("--connectionName", type=str, default='SQLITE_OPSIM', 
                       help="Key for the connection string to use in your dbLogin file -- "\
                            "Default is SQLITE_OPSIM")
    args = parser.parse_args()

    # Get db connection info.
    dbAddress = getData.getDbAddress(connectionName = args.connectionName)
    
    dbTable = args.simDataTable
    opsimrun = args.simDataTable.replace('output_', '')
    sessionID = opsimrun.split('_')[-1:][0]
    if args.propID == -666:
        args.propID = None
    
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
    fieldcols = ['fieldID', 'fieldRA', 'fieldDec']
    colnames = colnames + fieldcols
    colnames = list(set(colnames))
    
    # Get opsim simulation data
    simdata = getData.fetchSimData(dbTable, dbAddress, sqlconstraint, colnames)

    # Set up binner.
    fieldDataInfo = {}
    fieldDataInfo['useFieldTable'] = args.useFieldTable
    fieldDataInfo['dbAddress'] = dbAddress
    fieldDataInfo['fieldTable'] = args.fieldDataTable
    fieldDataInfo['sessionID'] = sessionID
    fieldDataInfo['proposalTable'] = args.proposalFieldTable
    fieldDataInfo['proposalID'] = args.propID
    bb = getBinner(simdata, fieldDataInfo)
    
    # Okay, go calculate the metrics.
    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"', '')
    gm = goBin(opsimrun, metadata, simdata, bb, metricList)

    # Generate some summary statistics and plots.
    printSummary(gm, metricList)
    # Generate (and save) plots.
    plot(gm)

    # Write the data to file.
    write(gm)
    
    gm.returnOutputFiles(verbose=True)
