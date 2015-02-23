# DOCUMENTATION & EXAMPLES -CM
#--------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db --ips 2
# ips:
#       The number of images to stitch together per second of video. If not specified, the default is 10.0.
# default - automatically sets movieStepsize to default of 365.25 days
# default - automatically sets fps to match ips = 2
# default - automatically sets outDir to default Output. Will create directory if not already present.
#
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db --ips 2 --cumulative
# cumulative:
#       This flag tells movie slicer to step though data to slice cumulatively, rather than bin.
#
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db --movieStepsize 20 --ips 6 --fps 6 --outDir myMovie --sqlConstraint "filter='g'"
#(because we specified a relatively low step size this will be more computationally intensive but result in more usable images for a video)
# movieStepsize:
#       Movie slicer will step through data with a stride of 180 days. Higher step size means fewer steps to reach the end of the
#       data, and fewer images. Conversely, a lower step size results in a greater number of images produced, and a greater number of calculations.
#       At stepSize=1 a calculation is being made at each day.
# fps:
#       Can specify a fps. If fps is higher than ips it will 'copy' images to meet the fps requirement. If fps is lower than ips, it will cut out
#       images from the sequence to maintain the fps. (note: it will still cover the same number of images in a second, it just has to choose which
#       ones it can display.) As a rule of thumb a fps>30 is pointless, it takes more time, and isn't detectable to the human eye.
# outDir:
#       Can specify an output directory. If it doesn't already exist, it will make one.
# filter:
#       Can specify a filter. In this case it will pull data in the 'g' band.
#
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db --skipComp --outDir myMovie --ips 4
# sc:
#       Flag to skip computation. Will make a movie out of the existing images in a directory.
#
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db -sc --outDir myMovie --movieLength 10
# (may want to specify movieStepsize smaller than default of 365.25, or it will be very choppy and discontinuous for long movieLengths)
# movieLength:
#       Can specify the length of the video in seconds. will automatically calculate corresponding ips & fps.
#
#-------------------------------------------------------------

# To modify the metrics calculated and used in the movie, edit the 'setupMetrics' section. The columns required from the
# database will be propagated to getData automatically. 


import os, argparse
import warnings
import fnmatch
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
from lsst.sims.maf.utils import ColInfo


def dtime(time_prev):
   return (time.time() - time_prev, time.time())


def setupMetrics(args, verbose=False):
    # Define and set up metrics.
    # Note that it is useful to set up the plotDict so that the min/max range for the plot
    #  is the same for all movie frames.
    t = time.time()
    metricList = []
    #Simple metrics: coadded depth and number of visits
    nvisitsMin = 0
    nvisitsMax = 300
    coaddMin = 25
    coaddMax = 28
    if not args.cumulative:
        # Take a guess ... probably will need to be adjusted for your stepsize.
        nvisitsMax = 15
        coaddMin = 24.0
        coaddMax = 26.5
    metricList.append(metrics.Coaddm5Metric('fiveSigmaDepth', metricName='Coaddm5Metric',
                                            plotDict={'colorMin':coaddMin, 'colorMax':coaddMax}))
    metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
                                            plotDict={'colorMin':nvisitsMin, 'colorMax':nvisitsMax,
                                                        'cbarFormat': '%d', 'title':'Number of Visits '}))
    dt, t = dtime(t)
    if verbose:
        print 'Set up metrics %f s' %(dt)
    return metricList

def getData(opsDb, sqlconstraint, metricList, args):
    # Find the columns required  by the metrics and slicers (including if they come from stackers).
    colInfo = ColInfo()
    dbcolnames = set()
    defaultstackers = set()
    # Look for the source of ra/dec columns.
    for col in args.raCol, args.decCol:
        colsource = colInfo.getDataSource(col)
        if colsource != colInfo.defaultDataSource:
            defaultstackers.add(colsource)
        else:
            dbcolnames.add(col)
    # Look for the source of columns in the metrics.
    for col in metricList[0].colRegistry.colSet:
        colsource = colInfo.getDataSource(col)
        if colsource != colInfo.defaultDataSource:
            defaultstackers.add(colsource)
        else:
            dbcolnames.add(col)
    # If you want to specify options for stackers, instante them and add to the list here.
    mystackers = []
    # Remove explicity instantiated stackers from defaultstacker set.
    for s in mystackers:
        if s.__class__ in defaultstackers:
            defaultstackers.remove(s.__class__)
    # Instantiate the remaining default stackers.
    for s in defaultstackers:
        mystackers.append(s())
    # Add the columns needed by stackers to the list to grab from the database.
    for s in mystackers:
        for col in s.colsReq:
            dbcolnames.add(col)
    # Add the expMJD for the movieslicer itself.
    dbcolnames.add('expMJD')
    # Get data from database.
    simdata = opsDb.fetchMetricData(dbcolnames, sqlconstraint)
    if len(simdata) == 0:
        raise Exception('No simdata found matching constraint %s' %(sqlconstraint))
    # Now add the stacker columns.
    for s in mystackers:
        simdata = s.run(simdata)
    return simdata

def setupMovieSlicer(simdata, binsize = 365.0, cumulative=True, verbose=False):
    t = time.time()
    ms = slicers.MovieSlicer(sliceColName='expMJD', binsize=binsize, cumulative=cumulative)
    ms.setupSlicer(simdata)
    dt, t = dtime(t)
    if verbose:
        print 'Set up movie slicer in %f s' %(dt)
    return ms

def setupHealpixSlicer(simdatasubset, racol, deccol, nside, verbose=False):
    t = time.time()
    hs = slicers.HealpixSlicer(nside=nside, spatialkey1=racol, spatialkey2=deccol, plotFuncs='plotSkyMap')
    hs.setupSlicer(simdatasubset)
    dt, t = dtime(t)
    if verbose:
        print 'Set up healpix slicer and built kdtree %f s' %(dt)
    return hs

def runSlices(opsimName, metadata, simdata, metricList, args, verbose=False):
    """Do the work to run the movie slicer, and at each step, setup the healpix slicer and run the metrics,
    making the plots."""
    # Set up movie slicer
    movieslicer = setupMovieSlicer(simdata, binsize = args.movieStepsize, cumulative=args.cumulative)
    start_date = movieslicer[0]['slicePoint']['binLeft']
    sliceformat = '%s0%dd' %('%', int(np.log10(len(movieslicer)))+1)
    # Run through the movie slicer slicePoints:
    for i, movieslice in enumerate(movieslicer):
        t = time.time()
        timeInterval = '%.2f to %.2f' %(movieslice['slicePoint']['binLeft']-start_date,
                                        movieslice['slicePoint']['binRight']-start_date)
        if verbose:
            print 'working on time interval %s' %(timeInterval)
        slicenumber = sliceformat %(i)
        for metric in metricList:
            metric.plotDict['label'] = 'Time: \n' + timeInterval + '\n' + args.sqlConstraint
        # Identify the subset of simdata in the movieslicer 'data slice'
        simdatasubset = simdata[movieslice['idxs']]

        # Set up healpix slicer on subset of simdata provided by movieslicer
        hs = setupHealpixSlicer(simdatasubset, args.raCol, args.decCol, args.nside)
        # Set up sliceMetric to handle healpix slicer + metrics calculation + plotting
        sm = sliceMetrics.RunSliceMetric(outDir=args.outDir, useResultsDb=False,
                                            figformat='png', dpi=72, thumbnail=False)
        sm.setSlicer(hs)
        sm.setMetrics(metricList)
        # Calculate metric data values for simdatasubset
        sm.runSlices(simdatasubset, simDataName=opsimName)
        # Plot data for this slice of the movie, adding slicenumber as a suffix for output plots
        sm.plotAll(outfileSuffix=slicenumber, closefig=True)
        # Write the data -- uncomment if you want to do this.
        # sm.writeAll(outfileSuffix=slicenumber)
        if verbose:
            dt, t = dtime(t)
            print 'Ran and plotted slice %s of movieslicer in %f s' %(slicenumber, dt)


def stitchMovie(metricList, args):
    # Create a movie slicer to access the movie generation routine.
    movieslicer = slicers.MovieSlicer()
    # Identify roots of distinct output plot files.
    outfileroots = []
    for metric in metricList:
        mName = metric.name.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        dbName = args.opsimDb.replace('_sqlite.db', '')
        outfileroots.append(dbName + '_' + mName + '_' + 'HEAL')

    for outfileroot in outfileroots:
        # Identify filenames.
        plotfiles = fnmatch.filter(os.listdir(args.outDir), outfileroot + '*SkyMap.png')
        slicenum = plotfiles[0].replace(outfileroot, '').replace('_SkyMap.png', '').replace('_', '')
        sliceformat = '%s0%dd' %('%', len(slicenum))
        n_images = len(plotfiles)
        if n_images == 0:
            raise Exception('No images found in %s with name like %s' %(args.outDir, outfileroot))
        # Set up ffmpeg FPS/IPS parameters.
        # If a movieLength was specified... set args.ips/fps according to the number of images.
        if args.movieLength != 0.0:
            #calculate images/second rate
            args.ips = n_images/float(args.movieLength)
            print "For a movie length of " + str(args.movieLength) + " IPS set to: ", args.ips
        if args.fps == 0.0:
            warnings.warn('(FPS of 0.0) Setting fps equal to ips, up to a value of 30fps.')
            if args.ips <= 30.0:
                args.fps = args.ips
            else:
                args.fps = 30.0
        if args.fps < args.ips:
            warnings.warn('Will create movie, but FPS < IPS, so some frames may be skipped.')
        if args.fps > 30.0:
            warnings.warn('Will create movie, but FPS above 30 reduces performance and is undetectable to the human eye.')
        # Create the movie.
        movieslicer.plotMovie(outfileroot, sliceformat, plotType='SkyMap', figformat='png',
                                outDir=args.outDir, ips=args.ips, fps=args.fps)


if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Opsim sqlite db file")
    parser.add_argument("--dbDir", type=str, default='.',
                        help="Directory containing opsim sqlite db file")
    parser.add_argument("--skipComp", action = 'store_true', default=False,
                        help="Just make movie from existing metric plot files.")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--movieStepsize", type=float, default=365., help="Step size for movie slicer. Default 365 (1 year).")
    parser.add_argument("--nside", type=int, default=128,
                        help="NSIDE parameter for healpix grid resolution. Default 128.")
    parser.add_argument("--raCol", type=str, default='fieldRA',
                        help="Name of RA column (fieldRA / nondithered is default).")
    parser.add_argument("--decCol", type=str, default='fieldDec',
                        help="Name of Dec column (fieldDec / nondithered is default).")
    parser.add_argument("--binned", action = 'store_true', default=False, help="Create binned, non-cumulative movie.")
    parser.add_argument("--outDir", type=str, default='Output', help="Output directory.")
    parser.add_argument("--ips", type=float, default = 10.0,
                        help="The number of images per second in the movie. Will skip accordingly if fps is lower.")
    parser.add_argument("--fps", type=float, default = 0.0, help="The frames per second of the movie.")
    parser.add_argument("--movieLength", type=float, default=0.0,
                        help="Enter the desired length of the movie in seconds. "
                        "If you do so, there is no need to enter images per second, it will be calculated.")

    args = parser.parse_args()

    # Flip the flag to the more intuitive value within the code. However, we anticipate most movies will be cumulative,
    #  so the flag is flipped in the argument list.
    args.cumulative = not args.binned

    start_t = time.time()

    # Check if directory exists; create if appropriate.
    if not os.path.isdir(args.outDir):
        if args.skipComp:
            raise Exception('Skipping metric generation, expect to find plots in %s directory but it does not exist.'
                            %(args.outDir))
        else:
            os.mkdir(args.outDir)

    # Check if user passed directory + filename as opsimDb.
    if len(os.path.dirname(args.opsimDb)) > 0:
        raise Exception('OpsimDB should be just the filename of the sqlite file (not %s). Use --dbDir.' %(args.opsimDb))

    # Set up metrics.
    metricList = setupMetrics(args)

    if not args.skipComp:
        # Get db connection info, and connect to database.
        opsimName =  args.opsimDb.replace('_sqlite.db', '')
        dbAddress = 'sqlite:///' + os.path.join(args.dbDir, args.opsimDb)
        opsDb = db.OpsimDatabase(dbAddress)
        sqlconstraint = args.sqlConstraint
        metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')
        # Get data from database.
        simdata = getData(opsDb, sqlconstraint, metricList, args)
        # Run the movie slicer (and at each step, healpix slicer and calculate metrics).
        gm = runSlices(opsimName, metadata, simdata, metricList, args)

    # Build movie.
    stitchMovie(metricList, args)

    end_t, start_t = dtime(start_t)
    print 'Total time: ', end_t
