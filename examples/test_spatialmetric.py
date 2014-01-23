import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics
import glob

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())



def getData(dbTable, dbAddress, bandpass):
    t = time.time()
    table = db.Table(dbTable, 'obsHistID', dbAddress)
    simdata = table.query_columns_RecArray(chunk_size=10000000, 
                                           constraint="filter = \'%s\'" %(bandpass), 
                                           colnames=['filter', 'expMJD',  'night',
                                                     'fieldRA', 'fieldDec', 'airmass',
                                                     '5sigma_modified', 'seeing',
                                                     'skybrightness_modified', 'altitude',
                                                     'hexdithra', 'hexdithdec'], 
                                                     groupByCol='expMJD')
    dt, t = dtime(t)
    print 'Query complete: %f s' %(dt)
    print 'Retrieved %d observations' %(len(simdata['expMJD']))
    return simdata

def getBinner(racol, deccol, nside=128):
    t = time.time()
    bb = binners.HealpixBinner(nside=nside)    
    bb.setupBinner(racol, deccol)
    
    dt, t = dtime(t)
    print 'Set up binner (and built kdtree if spatial binner) %f s' %(dt)
    return bb

def getMetrics():
    t = time.time()
    # Set up metrics.
    dtmin = 1./60./24.
    dtmax = 360./60./24.
    visitPairs = metrics.VisitPairsMetric(deltaTmin=dtmin, deltaTmax=dtmax)

    seeingcol = 'finSeeing' # if new v3 opsim
    seeingcol = 'seeing' # if old opsim (output_opsim3_61)
    meanseeing = metrics.MeanMetric(seeingcol)
    medianairmass = metrics.MedianMetric('airmass')
    minairmass = metrics.MinMetric('airmass')
    meanm5 = metrics.MeanMetric('5sigma_modified')
    maxm5 = metrics.MaxMetric('5sigma_modified')
    rmsm5 = metrics.RmsMetric('5sigma_modified')
    meanskybright = metrics.MeanMetric('skybrightness_modified')
    maxskybright = metrics.MaxMetric('skybrightness_modified')
    coaddm5 = metrics.Coaddm5Metric('5sigma_modified')
    nvisits = metrics.CountMetric(seeingcol)
    
    metricList = [meanseeing, medianairmass, meanm5, meanskybright, coaddm5, nvisits, visitPairs]

    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)

    return metricList


def goBin(dbTable, metadata, simdata, bb, metricList):
    t = time.time()
    gm = binMetrics.BaseBinMetric()
    gm.setBinner(bb)
    
    dt, t = dtime(t)
    print 'Set up gridMetric %f s' %(dt)

    gm.runBins(metricList, simdata, simDataName=dbTable, metadata = metadata)
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
            mean = gm.computeSummaryStatistics(m.name, metrics.MeanMetric)
            print 'SummaryNumber for', m.name, ':', mean
        except ValueError:
            pass
    dt, t = dtime(t)
    print 'Computed summaries %f s' %(dt)

if __name__ == '__main__':

    
    #dbTable = 'output_opsim3_61_forLynne' 
    #dbTable = 'output_opsim2_145_forLynne'   
    #dbAddress = 'mssql+pymssql://LSST-2:L$$TUser@fatboy.npl.washington.edu:1433/LSST' 
    #dbTable = 'output_opsimblitz2_1007'
    dbTable = 'output_opsim3_61'
    dbAddress = 'mysql://lsst:lsst@localhost/opsim?unix_socket=/opt/local/var/run/mariadb/mysqld.sock'
    bandpass = 'r'

    nside=128
        
    simdata = getData(dbTable, dbAddress, bandpass)
    bb = getBinner(simdata['fieldRA'], simdata['fieldDec'], nside=nside)
    metricList = getMetrics()

    gm = goBin(dbTable, bandpass, simdata, bb, metricList)
    printSummary(gm, metricList)
    
    plot(gm)
    write(gm)
    
    print 'Round 2 (dithered)'
    
    bandpass = 'r'
    #dbTable = 'output_opsimblitz2_1007'
    dbTable = 'output_opsim3_61'

    simdata = getData(dbTable, dbAddress, bandpass)
    bb = getBinner(simdata['hexdithra'], simdata['hexdithdec'], nside=nside)
    metricList = getMetrics()

    gm = goBin(dbTable, bandpass + ' dithered', simdata, bb, metricList)
    printSummary(gm, metricList)
    
    plot(gm)
    write(gm)

