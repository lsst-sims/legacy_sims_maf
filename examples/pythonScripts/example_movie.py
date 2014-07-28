## EXAMPLE
# python example_movie.py [sqlite db file]  (can be in another directory, just give relative path).

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


def setupMovieSlicer(simdata):
    t = time.time()
    ms = slicers.MovieSlicer(sliceColName='expMJD', binsize=365.25)
    ms.setupSlicer(simdata)
    dt, t = dtime(t)
    print 'Set up movie slicer in %f s' %(dt)
    return ms

def setupHealpixSlicer(simdatasubset, racol, deccol, nside):
    t = time.time()
    hs = slicers.HealpixSlicer(nside=nside, spatialkey1=racol, spatialkey2=deccol)    
    hs.setupSlicer(simdatasubset)
    dt, t = dtime(t)
    print 'Set up healpix slicer and built kdtree %f s' %(dt)
    return hs

    
def setupMetrics():
    # Set up metrics.
    t = time.time()
    metricList = []
    # Simple metrics: coadded depth and number of visits
    metricList.append(metrics.Coaddm5Metric('fiveSigmaDepth', 
                                            plotParams={'colorMin':25, 'colorMax':28}))
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                          plotParams={'logScale':False, 'title':'Number of visits',
                                                      'colorMin':0, 'colorMax':300,
                                                      'cbarFormat': '%d'}))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList


def run(opsimName, metadata, simdata, metricList, nside):
    """Do the work to run the movie slicer, and at each step, setup the healpix slicer and run the metrics,
    making the plots."""

    # Set up movie slicer
    movieslicer = setupMovieSlicer(simdata)

    # Ideally you'd translate the length of the movieslicer into the format statement for the
    #  movie slicer suffix (i.e. len(movieslicer) -> format = '%s.%dd' %('%', 2))

    # Run through the movie slicer slicePoints:
    for i, movieslice in enumerate(movieslicer):
        t = time.time()        
        slicenumber = '%.3d' %(i)
        # Identify the subset of simdata in the movieslicer 'data slice'
        simdatasubset = simdata[movieslice['idxs']]
        # Set up healpix slicer on subset of simdata provided by movieslicer
        hs = setupHealpixSlicer(simdatasubset, 'fieldRA', 'fieldDec', nside=nside)
        # Set up sliceMetric to handle healpix slicer + metrics calculation + plotting
        sm = sliceMetrics.RunSliceMetric()
        sm.setSlicer(hs)
        sm.setMetrics(metricList)
        # Calculate metric data values for simdatasubset
        sm.runSlices(simdatasubset, simDataName=opsimName, metadata=metadata)
        # Plot data for this slice of the movie, adding slicenumber as a suffix for output plots
        sm.plotAll(outfileSuffix=slicenumber, closefig=True)
        # Write the data -- uncomment if you want to do this.
        # sm.writeAll(outfileSuffix=slicenumber)
        
        dt, t = dtime(t)
        print 'Ran and plotted slice %s of movieslicer in %f s' %(slicenumber, dt)

    # Create the movie.
    # Chris - you need to write this method on movieSlicer and decide what arguments are necessary.
    # (e.g. do you pass slicenumbers back into plotMovie or are they generated & stored in movieslicer itself?)
    movieslicer.plotMovie()
    


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Filename for opsim sqlite db file")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--nside", type=int, default=64,
                        help="NSIDE parameter for healpix grid resolution. Default 64.")
    args = parser.parse_args()
    
    # Get db connection info, and connect to database.
    dbAddress = 'sqlite:///' + args.opsimDb
    oo = db.OpsimDatabase(dbAddress)
    opsimName = oo.fetchOpsimRunName()
    sqlconstraint = args.sqlConstraint
        
    # Set up metrics. 
    metricList = setupMetrics()

    # Find columns that are required by metrics.
    colnames = list(metricList[0].colRegistry.colSet)
    # Add columns needed for healpix slicer.
    fieldcols = ['fieldRA', 'fieldDec']
    colnames += fieldcols
    # Add column needed for movie slicer.
    moviecol = ['expMJD',]
    colnames += moviecol
    # Remove duplicates.
    colnames = list(set(colnames))
    
    # Get data from database.
    simdata = oo.fetchMetricData(colnames, sqlconstraint)
    
    # Run the movie slicer (and at each step, healpix slicer and calculate metrics).
    comment = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')
    gm = run(opsimName, comment, simdata, metricList, args.nside)

    
