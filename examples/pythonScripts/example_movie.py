# DOCUMENTATION & EXAMPLES -CM
#--------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db --ips 2
# ips:
#       The number of images to stitch together per second of video. If not specified, the default is 10.0.
# default - automatically sets movieStepsize to default of 365.25 days
# default - automatically sets fps to match ips = 2
# default - automatically sets outDir to default Output. Will create directory if not already present.
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db -bin --ips 2
# bin:
#       This flag tells movie slicer to step though data bin by bin rather than slicing cumulatively.
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
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db -sc --outDir myMovie --ips 4
# sc:
#       Flag to skip computation. Will make a movie out of the existing images in a directory.
#-------------------------------------------------------------
# python example_movie.py opsimblitz2_1060_sqlite.db -sc --outDir myMovie --movieLength 10
# (may want to specify movieStepsize smaller than default of 365.25, or it will be very choppy and discontinuous for long movieLengths)
# movieLength:
#       Can specify the length of the video in seconds. will automatically calculate corresponding ips & fps.
#-------------------------------------------------------------
import os, argparse
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import time
import warnings
import fnmatch


def dtime(time_prev):
   return (time.time() - time_prev, time.time())


def setupMovieSlicer(simdata, binsize = 365.25, cumulative=True):
    t = time.time()
    ms = slicers.MovieSlicer(sliceColName='expMJD', binsize=binsize, cumulative=cumulative)
    ms.setupSlicer(simdata)
    dt, t = dtime(t)
    print 'Set up movie slicer in %f s' %(dt)
    return ms

def setupHealpixSlicer(simdatasubset, racol, deccol, nside):
    t = time.time()
    hs = slicers.HealpixSlicer(nside=nside, spatialkey1=racol, spatialkey2=deccol, plotFuncs='plotSkyMap')
    hs.setupSlicer(simdatasubset)
    dt, t = dtime(t)
    print 'Set up healpix slicer and built kdtree %f s' %(dt)
    return hs


def setupMetrics():
    # Set up metrics.
    t = time.time()
    metricList = []
    #Simple metrics: coadded depth and number of visits
    # metricList.append(metrics.Coaddm5Metric('fiveSigmaDepth', metricName='Coaddm5Metric',
    #                                          plotDict={'colorMin':25, 'colorMax':28}))
    # metricList.append(metrics.CountMetric('expMJD', metricName='N_Visits',
    #                                        plotDict={'logScale':False,
    #                                                    'colorMin':0, 'colorMax':320,
    #                                                    'cbarFormat': '%d', 'title':'Number of Visits '}))
    # metricList.append(metrics.SumMetric('expMJD', metricName='Sum',
    #                                       plotDict={'logScale':True,
    #                                                   'cbarFormat': '%d', 'title':'Sum Metric'}))
    metricList.append(metrics.CountMetric('expMJD', metricName='Number of Visits',
                                          plotDict={'logScale':False,
                                                      'colorMin':0, 'colorMax':300,
                                                      'cbarFormat': '%d'}))
    dt, t = dtime(t)
    print 'Set up metrics %f s' %(dt)
    return metricList


def run(opsimName, metadata, simdata, metricList, args):
    """Do the work to run the movie slicer, and at each step, setup the healpix slicer and run the metrics,
    making the plots."""

    sliceformat = '%04d'

    #calculation of metrics
    if args.skipComp:
        # Set up movie slicer
        movieslicer = setupMovieSlicer(simdata, binsize = args.movieStepsize, cumulative=args.cumulative)
        start_date = movieslicer[0]['slicePoint']['binLeft']
        # Run through the movie slicer slicePoints:
        for i, movieslice in enumerate(movieslicer):
            t = time.time()
            timeInterval = '%.2f to %.2f' %(movieslice['slicePoint']['binLeft']-start_date, movieslice['slicePoint']['binRight']-start_date)
            print 'working on time interval %s' %(timeInterval)
            slicenumber = '%.4d' %(i)

            #adding day number to title of plot
            if args.cumulative:
                for metric in metricList:
                    metric.plotDict['label'] = 'day: ' + str(i*args.movieStepsize) + '\n' + args.sqlConstraint
            else:
                for metric in metricList:
                    metric.plotDict['label'] = 'time interval: ' + '\n' + timeInterval + '\n'+ '\n' + args.sqlConstraint

            # Identify the subset of simdata in the movieslicer 'data slice'
            simdatasubset = simdata[movieslice['idxs']]
            # Set up healpix slicer on subset of simdata provided by movieslicer
            hs = setupHealpixSlicer(simdatasubset, 'ditheredRA', 'ditheredDec', args.nside)
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

            dt, t = dtime(t)
            print 'Ran and plotted slice %s of movieslicer in %f s' %(slicenumber, dt)
    else:
        #create movieslicer to call plotMovie
        movieslicer = slicers.MovieSlicer()

    outfileroots = []
    for metric in metricList:
        #Create plot names
        Mname = metric.name.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        dbName = args.opsimDb.strip('_sqlite.db')
        outfileroots.append(dbName + '_' + Mname + '_' + 'HEAL')

    #if a movieLength was specified
    if args.movieLength != 0.0:
        if args.skipComp:
            args.ips = len(movieslicer)/args.movieLength
        else:
            #figure out how many images there are.
            n_images = len(fnmatch.filter(os.listdir(args.outDir), outfileroots[0] + '*SkyMap.png'))
            print outfileroots[0]
            if n_images == 0:
                warnings.warn('Targeted files did not match directory contents. Make sure the parameters of this run, match the files. (for instance metrics)')
            #calculate images/second rate
            args.ips = n_images/args.movieLength
        print "for a movie length of " + str(args.movieLength) + " IPS set to: ", args.ips

    if args.fps == 0.0:
            warnings.warn('(FPS of 0.0) Setting fps equal to ips, up to a value of 30fps.')
            if args.ips <= 30.0:
                args.fps = args.ips
            else:
                args.fps = 30.0
    # Create the movie for every metric that was run
    for outroot in outfileroots:
       out = outroot.split('/')[-1]
       movieslicer.plotMovie(out, sliceformat, plotType='SkyMap', figformat='png', outDir=args.outDir, ips=args.ips, fps=args.fps)

if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Filename for opsim sqlite db file")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--movieStepsize", type=float, default=365.25, help="Step size for movie slicer.")
    parser.add_argument("--nside", type=int, default=64,
                        help="NSIDE parameter for healpix grid resolution. Default 64.")
    parser.add_argument("--outDir", type=str, default='Output', help="Output directory.")
    parser.add_argument("-bin", "--cumulative", action = 'store_false', default=True, help="Create cumulative movie, or bin?")
    parser.add_argument("--ips", type=float, default = 10.0,
                        help="The number of images per second in the movie. Will skip accordingly if fps is lower.")
    parser.add_argument("--fps", type=float, default = 0.0, help="The frames per second of the movie.")
    parser.add_argument("-sc", "--skipComp", action = 'store_false', default=True,
                        help="Compute the metrics(True), or just make movie from existing files(False).")
    parser.add_argument("--movieLength", type=float, default=0.0,
                        help="Enter the desired length of the movie in seconds. If you do so, there is no need to enter images per second, it will be calculated.")
    args = parser.parse_args()
    start_t = time.time()
    #cleaning up movie parameters
    if args.fps > 30.0:
        warnings.warn('FPS above 30 reduces performance and is undetectable to the human eye. Try lowering the fps.')
    if not os.path.isdir(args.outDir):
         warnings.warn('Cannot find output directory named %s, creating directory.' %(args.outDir))
         os.mkdir(args.outDir)

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
    fieldcols = ['fieldRA', 'fieldDec', 'ditheredRA', 'ditheredDec']
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
    ##for goodness sake, why not just pass it args itself?
    gm = run(opsimName, comment, simdata, metricList, args)
    end_t, start_t = dtime(start_t)
    print 'Total time: ', end_t
