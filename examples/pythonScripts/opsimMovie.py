# python $SIMS_MAF_DIR/examples/pythonScripts/opsimMovie.py ops2_1088_sqlite.db --sqlConstraint 'night=130' --outDir Output
#
# --ips = number of images to stitch together per second of view (default is 10).
# --fps = frames per second for the output video .. default just matches ips. If specified as higher than ips,
#         the additional frames will be copied to meet the fps requirement. If fps is lower than ips, images will be
#         removed from the sequence to maintain fps. As a rule of thumb, fps>30 is undetectable to the human eye.
# --movieLength = can specify the length of the output video (in seconds), then automatically calculate corresponding ips
#         and fps based on the number of movie slices. 
# --skipComp = skip computing the metrics and generating plots, just use files from disk.
#

import os, argparse
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.sliceMetrics as sliceMetrics

import time
import warnings
import fnmatch

lat_tele = np.radians(-29.666667)

def dtime(time_prev):
    return (time.time() - time_prev, time.time())


def getData(opsDb, sqlconstraint):
    # Define columns we want from opsim database (for metrics, slicers and stacker info).
    colnames = ['expMJD', 'filter', 'fieldID', 'fieldRA', 'fieldDec', 'lst',
                'moonRA', 'moonDec', 'moonPhase']
    # Get data from database.
    simdata = opsDb.fetchMetricData(colnames, sqlconstraint)
    if len(simdata) == 0:
        raise Exception('No simdata found matching constraint %s' %(sqlconstraint))
    # Add stacker columns.
    hourangleStacker = stackers.HourAngleStacker()
    simdata = hourangleStacker.run(simdata)
    filterStacker = stackers.FilterColorStacker()
    simdata = filterStacker.run(simdata)
    # Fetch field data.
    fields = opsDb.fetchFieldsFromFieldTable()
    return simdata, fields

def setupMetrics(opsimName, metadata, visittime, verbose=False):
    # Set up metrics. Will apply one to ms and one to ms_curr.
    t = time.time()
    import matplotlib.cm as cm
    metric = metrics.VisitFiltersMetric(t0=visittime)
    metric.plotDict['title'] = "%s: %s" %(opsimName, metadata)
    dt, t = dtime(t)
    if verbose:
        print 'Set up metrics %f s' %(dt)
    return metric

def setupMovieSlicer(simdata, bins, verbose=False):
    t = time.time()
    movieslicer = slicers.MovieSlicer(sliceColName='expMJD', bins=bins, cumulative=True)
    movieslicer.setupSlicer(simdata)
    dt, t = dtime(t)
    if verbose:
        print 'Set up movie slicers in %f s' %(dt)
    return movieslicer

def setupOpsimFieldSlicer(simdatasubset, fields, verbose=False):
    t = time.time()
    ops = slicers.OpsimFieldSlicer(plotFuncs='plotSkyMap')
    ops.setupSlicer(simdatasubset, fields)
    dt, t = dtime(t)
    if verbose:
        print 'Set up opsim field slicer in %s' %(dt)
    return ops


def runSlices(opsimName, metadata, simdata, fields, bins, args, verbose=False):
    # Set up the movie slicer.
    movieslicer = setupMovieSlicer(simdata, bins)
    # Set up formatting for output suffix.
    sliceformat = '%s0%dd' %('%', int(np.log10(len(movieslicer)))+1)
    # Run through the movie slicer slicePoints and generate plots at each point.
    for i, ms in enumerate(movieslicer):
        t = time.time()
        slicenumber = sliceformat %(i)
        if verbose:
            print slicenumber
        # Set up metrics.
        metric = setupMetrics(opsimName, metadata, ms['slicePoint']['binRight'], verbose)
        # Add time to plot label.
        metric.plotDict['label'] = 'Time: %f' %(ms['slicePoint']['binRight'])
        # Identify the subset of simdata in the movieslicer 'data slice'
        simdatasubset = simdata[ms['idxs']]
        # Set up opsim slicer on subset of simdata provided by movieslicer
        ops = setupOpsimFieldSlicer(simdatasubset, fields)
        # Set up sliceMetric to handle healpix slicer + metrics calculation + plotting
        sm = sliceMetrics.RunSliceMetric(outDir = args.outDir, useResultsDb=False,
                                                figformat='png', dpi=72, thumbnail=False)
        sm.setSlicer(ops)
        sm.setMetrics([metric])
        sm.runSlices(simdatasubset, simDataName=opsimName)
        # Plot data for this slice of the movie, adding slicenumber as a suffix for output plots
        obsnow = np.where(simdatasubset['expMJD'] == simdatasubset['expMJD'].max())[0]
        raCen = np.mean(simdatasubset[obsnow]['lst'])
        fignum = ops.plotSkyMap(sm.metricValues[0], raCen=raCen, **sm.plotDicts[0])
        fig = plt.figure(fignum)
        ax = plt.gca()
        # Add a legend.
        filterstacker = stackers.FilterColorStacker()
        for i, f in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            plt.figtext(0.92, 0.55 - i*0.035, f, color=filterstacker.filter_rgb_map[f])
        # Add a moon.
        moonRA = np.mean(simdatasubset[obsnow]['moonRA'])
        lon = -(moonRA - raCen - np.pi) % (np.pi*2) - np.pi
        moonDec = np.mean(simdatasubset[obsnow]['moonDec'])
        # Note that moonphase is 0-100 (translate to 0-1). 0=new.
        moonPhase = np.mean(simdatasubset[obsnow]['moonPhase'])/100.
        alpha = np.max([moonPhase, 0.15])
        circle = Circle((lon, moonDec), radius=0.05, color='k', alpha=alpha)
        ax.add_patch(circle)
        # Add horizon and zenith.
        plt.plot(0, lat_tele, 'k+')
        step = 0.002
        theta = np.arange(0, np.pi*2 +step/2., step)
        rad = np.radians(90.)
        x = rad*np.sin(theta)
        y = rad*np.cos(theta) + lat_tele
        plt.plot(x, y, 'k-', alpha=0.3)
        plt.savefig(os.path.join(args.outDir, 'movieFrame_' + slicenumber + '_SkyMap.png'), format='png', dpi=72)
        plt.close('all')
        dt, t = dtime(t)
        if verbose:
            print 'Ran and plotted slice %s of movieslicer in %f s' %(slicenumber, dt)


def stitchMovie(args):
    # Create a movie slicer to access the movie generation routine.
    movieslicer = slicers.MovieSlicer()
    outfileroot = 'movieFrame'
    # Identify filenames.
    plotfiles = fnmatch.filter(os.listdir(args.outDir), outfileroot + '*SkyMap.png')
    slicenum = plotfiles[0].strip(outfileroot).strip('_SkyMap.png')
    sliceformat = '%s0%dd' %('%', len(slicenum))
    n_images = len(plotfiles)
    if n_images == 0:
        raise Exception('No images found in %s with name like %s' %(args.outDir, outfileroot))
    # Set up ffmpeg parameters.
    # If a movieLength was specified... set args.ips/fps.
    if args.movieLength != 0.0:
        #calculate images/second rate
        args.ips = n_images/args.movieLength
        print "for a movie length of " + str(args.movieLength) + " IPS set to: ", args.ips
    if args.fps == 0.0:
        warnings.warn('(FPS of 0.0) Setting fps equal to ips, up to a value of 30fps.')
        if args.ips <= 30.0:
            args.fps = args.ips
        else:
            args.fps = 30.0
    # Create the movie.
    movieslicer.plotMovie(outfileroot, sliceformat, plotType='SkyMap', figformat='png',
                          outDir=args.outDir, ips=args.ips, fps=args.fps)

if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Filename for opsim sqlite db file")
    parser.add_argument("--sqlConstraint", type=str, default="filter='r'",
                        help="SQL constraint, such as filter='r' or propID=182")
    parser.add_argument("--movieStepsize", type=float, default=0, help="Step size (in days) for movie slicer. "
                        "Default sets 1 visit = 1 step.")
    parser.add_argument("--outDir", type=str, default='Output', help="Output directory.")
    parser.add_argument("--addPreviousObs", action='store_true', default=False,
                        help="Add all previous observations into movie (as background).")
    parser.add_argument("--skipComp", action = 'store_true', default=False,
                        help="Just make movie from existing metric plot files (True).")
    parser.add_argument("--ips", type=float, default = 10.0,
                        help="The number of images per second in the movie. Will skip accordingly if fps is lower.")
    parser.add_argument("--fps", type=float, default = 0.0,
                        help="The frames per second of the movie.")
    parser.add_argument("--movieLength", type=float, default=0.0,
                        help="Enter the desired length of the movie in seconds. "
                        "If you do so, there is no need to enter images per second, it will be calculated.")
    args = parser.parse_args()

    start_t = time.time()
    #cleaning up movie parameters
    if args.fps > 30.0:
        warnings.warn('FPS above 30 reduces performance and is undetectable to the human eye. Try lowering the fps.')
    if not os.path.isdir(args.outDir):
        if args.skipComp:
            raise Exception('Skipping metric generation, expect to find plots in %s directory but it does not exist.'
                            %(args.outDir))
        else:
            os.mkdir(args.outDir)

    if not args.skipComp:
        # Get db connection info, and connect to database.
        dbAddress = 'sqlite:///' + args.opsimDb
        oo = db.OpsimDatabase(dbAddress)
        opsimName = oo.fetchOpsimRunName()
        sqlconstraint = args.sqlConstraint
        metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')
        # Fetch the data from opsim.
        simdata, fields = getData(oo, sqlconstraint)
        # Set up the time bins for the movie slicer.
        start_date = simdata['expMJD'][0]
        if args.movieStepsize == 0:
            bins = simdata['expMJD']
        else:
            end_date = simdata['expMJD'].max()
            bins = np.arange(start_date, end_date+args.movieStepSize/2.0, args.movieStepSize, float)
        if args.addPreviousObs:
            # Go back and grab all the data, including all previous observations.
            if "night =" in sqlconstraint:
                sqlconstraint = sqlconstraint.replace("night =", "night <=")
            elif "night=" in sqlconstraint:
                sqlconstraint = sqlconstraint.replace("night=", "night<=")
            simdata, fields = getData(oo, sqlconstraint)
            # Update the first bin to be prior to the earliest opsim time.
            bins[0] = simdata['expMJD'][0]

        # Run the movie slicer (and at each step, setup opsim slicer and calculate metrics).
        runSlices(opsimName, metadata, simdata, fields, bins, args)

    stitchMovie(args)
    end_t, start_t = dtime(start_t)
    print 'Total time to create movie: ', end_t
