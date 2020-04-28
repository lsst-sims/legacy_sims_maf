from __future__ import print_function
from builtins import zip
from builtins import str
# python $SIMS_MAF_DIR/examples/pythonScripts/opsimMovie.py enigma_1189_sqlite.db --sqlConstraint 'night=130' --outDir Output
#
# --ips = number of images to stitch together per second of view (default is 10).
# --fps = frames per second for the output video .. default just matches ips. If specified as higher than ips,
#         the additional frames will be copied to meet the fps requirement. If fps is lower than ips,
#         images will be removed from the sequence to maintain fps.
#         As a rule of thumb, fps>30 is undetectable to the human eye.
# --movieLength = can specify the length of the output video (in seconds), then automatically
#         calculate corresponding ips and fps based on the number of movie slices.
# --skipComp = skip computing the metrics and generating plots, just use files from disk.
#

import os, argparse
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import ephem

import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.plots as plots
from lsst.sims.utils import Site

import time
import warnings
import fnmatch


def dtime(time_prev):
    return (time.time() - time_prev, time.time())


def getData(opsDb, sqlconstraint):
    # Define columns we want from opsim database (for metrics, slicers and stacker info).
    colnames = ['observationStartMJD', 'filter', 'fieldRA', 'fieldDec', 'observationStartLST',
                'moonRA', 'moonDec', 'moonPhase', 'night']
    # Get data from database.
    simdata = opsDb.fetchMetricData(colnames, sqlconstraint)
    if len(simdata) == 0:
        raise Exception('No simdata found matching constraint %s' %(sqlconstraint))
    # Add stacker columns.
    hourangleStacker = stackers.HourAngleStacker()
    simdata = hourangleStacker.run(simdata)
    filterStacker = stackers.FilterColorStacker()
    simdata = filterStacker.run(simdata)
    fieldIdStacker = stackers.OpSimFieldStacker()
    simdata = fieldIdStacker.run(simdata)
    # Fetch field data.
    #fields = opsDb.fetchFieldsFromFieldTable(degreesToRadians=True)
    fidx = np.unique(simdata['opsimFieldId'], return_index=True)[1]
    fields = np.recarray(len(fidx), dtype=[('opsimFieldId', np.float), ('fieldRA', np.float),
                                           ('fieldDec', np.float)])
    fields['opsimFieldId'] = simdata['opsimFieldId'][fidx]
    fields['fieldRA'] = np.radians(simdata['fieldRA'][fidx])
    fields['fieldDec'] = np.radians(simdata['fieldDec'][fidx])
    return simdata, fields

def setupMetrics(opsimName, metadata, plotlabel='', t0=0, tStep=40./24./60./60., years=0,
                 onlyFilterColors=False, verbose=False):
    # Set up metrics and plotDicts (they will be bundled with the appropriate opsim slicers below).
    t = time.time()
    nvisitsMax = 90*(years+1)
    colorMax = int(nvisitsMax/4)
    metricList = []
    plotDictList = []
    figsize = (8, 6)
    if not onlyFilterColors:
        metricList.append(metrics.CountMetric('observationStartMJD', metricName='Nvisits'))
        plotDictList.append({'colorMin': 0, 'colorMax': nvisitsMax,
                             'xlabel': 'Number of visits', 'title': 'Cumulative visits (all bands)',
                             'label': plotlabel, 'metricIsColor': False, 'figsize': figsize})
        for f in (['u', 'g', 'r', 'i', 'z', 'y']):
            metricList.append(metrics.CountSubsetMetric('filter', subset=f, metricName='Nvisits_' + f))
            plotDictList.append({'colorMin': 0, 'colorMax': colorMax, 'cbarFormat': '%d',
                                 'xlabel': 'Number of Visits', 'title': '%s band' %(f),
                                 'label': plotlabel, 'metricIsColor': False, 'figsize': figsize})
    metricList.append(metrics.FilterColorsMetric(t0=t0, tStep=tStep))
    plotDictList.append({'title': 'Simulation %s: %s' %(opsimName, metadata), 'bgcolor': None,
                         'metricIsColor': True, 'figsize': figsize})
    dt, t = dtime(t)
    if verbose:
        print('Set up metrics %f s' %(dt))
    return metricList, plotDictList

def setupMovieSlicer(simdata, bins, verbose=False):
    t = time.time()
    movieslicer = slicers.MovieSlicer(sliceColName='observationStartMJD', bins=bins, cumulative=True)
    movieslicer.setupSlicer(simdata)
    dt, t = dtime(t)
    if verbose:
        print('Set up movie slicers in %f s' %(dt))
    return movieslicer

def addHorizon(horizon_altitude=np.radians(20.), lat_telescope=Site(name='LSST').latitude_rad):
    """
    Adds a horizon at horizon_altitude, using the telescope latitude lat_telescope.
    Returns the lon/lat points that would be appropriate to add to a SkyMap plot centered on raCen.
    """
    step = .02
    az = np.arange(0, np.pi*2.0+step, step)
    alt = np.ones(len(az), float) * horizon_altitude
    obs = ephem.Observer()
    obs.lat = lat_telescope
    # Set obs lon to zero, just to fix the location.
    # Note that this is not the true observatory longitude, but as long as
    #  we calculate the RA at zenith for this longitude, we can still calculate HA appropriately.
    obs.lon = 0
    obs.pressure=0
    # Given obs lon at zero, find the equivalent ra overhead.
    zenithra, zenithlat = obs.radec_of(0, 90)
    lon = np.zeros(len(az), float)
    lat = np.zeros(len(az), float)
    for i, (alti, azi) in enumerate(zip(alt, az)):
        # Find the equivalent ra/dec values for an alt/az circle.
        r, lat[i] = obs.radec_of(azi, alti)
        # Correct the ra value by the zenith ra value, to get the HA.
        lon[i] = r - zenithra
    lon = -(lon - np.pi) % (np.pi*2) - np.pi
    return lon, lat

def runSlices(opsimName, metadata, simdata, fields, bins, args, opsDb, verbose=False):
    # Set up the movie slicer.
    movieslicer = setupMovieSlicer(simdata, bins)
    # Set up formatting for output suffix.
    sliceformat = '%s0%dd' %('%', int(np.log10(len(movieslicer)))+1)
    # Get the telescope latitude info.
    lat_tele = Site(name='LSST').latitude_rad
    # Run through the movie slicer slicePoints and generate plots at each point.
    for i, ms in enumerate(movieslicer):
        t = time.time()
        slicenumber = sliceformat %(i)
        if verbose:
            print(slicenumber)
        # Set up metrics.
        if args.movieStepsize != 0:
            tstep = args.movieStepsize
        else:
            tstep = ms['slicePoint']['binRight'] - bins[i]
            if tstep > 1:
                tstep = 40./24./60./60.
        # Add simple view of time to plot label.
        times_from_start = ms['slicePoint']['binRight'] - (int(bins[0]) + 0.16 - 0.5)
        # Opsim years are 365 days (not 365.25)
        years = int(times_from_start/365)
        days = times_from_start - years*365
        plotlabel = 'Year %d Day %.4f' %(years, days)
        # Set up metrics.
        metricList, plotDictList = setupMetrics(opsimName, metadata, plotlabel=plotlabel,
                                                t0=ms['slicePoint']['binRight'], tStep=tstep,
                                                years=years, verbose=verbose)
        # Identify the subset of simdata in the movieslicer 'data slice'
        simdatasubset = simdata[ms['idxs']]
        # Set up opsim slicer on subset of simdata provided by movieslicer
        opslicer = slicers.OpsimFieldSlicer(simDataFieldIdColName='opsimFieldId',
                                            fieldIdColName='opsimFieldId')
        # Set up metricBundles to combine metrics, plotdicts and slicer.
        bundles = []
        sqlconstraint = ''
        for metric, plotDict in zip(metricList, plotDictList):
            bundles.append(metricBundles.MetricBundle(metric, opslicer, constraint=sqlconstraint,
                                                      metadata=metadata, runName=opsimName,
                                                      plotDict=plotDict))
        # Remove (default) stackers from bundles, because we've already run them above on the original data.
        for mb in bundles:
            mb.stackerList = []
        bundledict = metricBundles.makeBundlesDictFromList(bundles)
        # Set up metricBundleGroup to handle metrics calculation + plotting
        bg = metricBundles.MetricBundleGroup(bundledict, opsDb, outDir=args.outDir,
                                             resultsDb=None, saveEarly=False)
        # 'Hack' bundleGroup to just go ahead and run the metrics, without querying the database.
        simData = simdatasubset
        bg.fieldData = fields
        bg.setCurrent(sqlconstraint)
        bg.runCurrent(constraint = sqlconstraint, simData=simData)
        # Plot data each metric, for this slice of the movie, adding slicenumber as a suffix for output plots.
        # Plotting here, rather than automatically via sliceMetric method because we're going to rotate the sky,
        #  and add extra legend info and figure text (for FilterColors metric).
        ph = plots.PlotHandler(outDir=args.outDir, figformat='png', dpi=72, thumbnail=False, savefig=False)
        obsnow = np.where(simdatasubset['observationStartMJD'] ==
                          simdatasubset['observationStartMJD'].max())[0]
        raCen = np.radians(np.mean(simdatasubset[obsnow]['observationStartLST']))
        # Calculate horizon location.
        horizonlon, horizonlat = addHorizon(lat_telescope=lat_tele)
        # Create the plot for each metric and save it (after some additional manipulation).
        for mb in bundles:
            ph.setMetricBundles([mb])
            fignum = ph.plot(plotFunc=plots.BaseSkyMap(), plotDicts={'raCen':raCen})
            fig = plt.figure(fignum)
            ax = plt.gca()
            # Add horizon and zenith.
            plt.plot(horizonlon, horizonlat, 'k.', alpha=0.3, markersize=1.8)
            plt.plot(0, lat_tele, 'k+')
            # For the FilterColors metric, add some extra items.
            if mb.metric.name == 'FilterColors':
                # Add the time stamp info (plotlabel) with a fancybox.
                plt.figtext(0.75, 0.9, '%s' %(plotlabel), bbox=dict(boxstyle='Round, pad=0.7',
                                                                    fc='w', ec='k', alpha=0.5))
                # Add a legend for the filters.
                filterstacker = stackers.FilterColorStacker()
                for i, f in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
                    plt.figtext(0.92, 0.55 - i*0.035, f, color=filterstacker.filter_rgb_map[f])
                # Add a moon.
                moonRA = np.radians(np.mean(simdatasubset[obsnow]['moonRA']))
                lon = -(moonRA - raCen - np.pi) % (np.pi*2) - np.pi
                moonDec = np.radians(np.mean(simdatasubset[obsnow]['moonDec']))
                # Note that moonphase is 0-100 (translate to 0-1). 0=new.
                moonPhase = np.mean(simdatasubset[obsnow]['moonPhase'])/100.
                alpha = np.max([moonPhase, 0.15])
                circle = Circle((lon, moonDec), radius=0.05, color='k', alpha=alpha)
                ax.add_patch(circle)
                # Add some explanatory text.
                ecliptic = Line2D([], [], color='r', label="Ecliptic plane")
                galaxy = Line2D([], [], color='b', label="Galactic plane")
                horizon = Line2D([], [], color='k', alpha=0.3, label="20 deg elevation limit")
                moon = Line2D([], [], color='k', linestyle='', marker='o', markersize=8, alpha=alpha,
                              label="\nMoon (Dark=Full)\n         (Light=New)")
                zenith = Line2D([], [], color='k', linestyle='', marker='+', markersize=5, label="Zenith")
                plt.legend(handles=[horizon, zenith, galaxy, ecliptic, moon], loc=[0.1, -0.35],
                           ncol=3, frameon=False,
                    title = 'Aitoff plot showing HA/Dec of simulated survey pointings',
                           numpoints=1, fontsize='small')
            # Save figure.
            plt.savefig(os.path.join(args.outDir, mb.metric.name + '_' + slicenumber + '_SkyMap.png'),
                        format='png', dpi=72)
            plt.close('all')
            dt, t = dtime(t)
            if verbose:
                print('Ran and plotted slice %s of movieslicer in %f s' %(slicenumber, dt))


def stitchMovie(metricList, args):
    # Create a movie slicer to access the movie generation routine.
    movieslicer = slicers.MovieSlicer()
    for metric in metricList:
        # Identify filenames.
        outfileroot = metric.name
        plotfiles = fnmatch.filter(os.listdir(args.outDir), outfileroot + '*_SkyMap.png')
        slicenum = plotfiles[0].replace(outfileroot, '').replace('_SkyMap.png', '').replace('_', '')
        sliceformat = '%s0%dd' %('%', len(slicenum))
        n_images = len(plotfiles)
        if n_images == 0:
            raise Exception('No images found in %s with name like %s' %(args.outDir, outfileroot))
        # Set up ffmpeg parameters.
        # If a movieLength was specified... set args.ips/fps.
        if args.movieLength != 0.0:
            #calculate images/second rate
            args.ips = int(n_images/args.movieLength)
            print("for a movie length of " + str(args.movieLength) + " IPS set to: ", args.ips)
        if args.fps == 0:
            warnings.warn('(FPS of 0) Setting fps equal to ips, up to a value of 30fps.')
            if args.ips <= 30:
                args.fps = args.ips
            else:
                args.fps = 30
        # Create the movie.
        movieslicer.makeMovie(outfileroot, sliceformat, plotType='SkyMap', figformat='png',
                                outDir=args.outDir, ips=args.ips, fps=args.fps)

if __name__ == '__main__':

    # Parse command line arguments for database connection info.
    parser = argparse.ArgumentParser()
    parser.add_argument("opsimDb", type=str, help="Opsim sqlite db file")
    parser.add_argument("--dbDir", type=str, default='.',
                        help="Directory containing opsim sqlite db file")
    parser.add_argument("--sqlConstraint", type=str, default=None,
                        help="SQL constraint, such as filter='r' or night=1500. Default None.")
    parser.add_argument("--movieStepsize", type=float, default=0, help="Step size (in days) for movie slicer. "
                        "Default sets 1 visit = 1 step.")
    parser.add_argument("--outDir", type=str, default='Output', help="Output directory.")
    parser.add_argument("--addPreviousObs", action='store_true', default=False,
                        help="Add all previous observations into movie (as background).")
    parser.add_argument("--skipComp", action = 'store_true', default=False,
                        help="Just make movie from existing metric plot files.")
    parser.add_argument("--ips", type=float, default = 30,
                        help="The number of images per second in the movie. "
                             "Will skip accordingly if fps is lower. Default 30.")
    parser.add_argument("--fps", type=float, default = 30,
                        help="The frames per second of the movie. Default 30.")
    parser.add_argument("--movieLength", type=float, default=0.0,
                        help="Enter the desired length of the movie in seconds. "
                        "If you do so, there is no need to enter images per second, it will be calculated.")
    args = parser.parse_args()

    start_t = time.time()
    #cleaning up movie parameters
    if args.fps > 30:
        warnings.warn('FPS above 30 reduces performance and is undetectable to the human eye. Try lowering the fps.')

    # Check if output directory exists; create if appropriate.
    if not os.path.isdir(args.outDir):
        if args.skipComp:
            raise Exception('Skipping metric generation, expect to find plots in %s directory but it does not exist.'
                            %(args.outDir))
        else:
            os.mkdir(args.outDir)

    # Check if user passed directory + filename as opsimDb.
    if len(os.path.dirname(args.opsimDb)) > 0:
        raise Exception('OpsimDB should be just the filename of the sqlite file (not %s). Use --dbDir.' %(args.opsimDb))

    opsimName =  args.opsimDb.replace('_sqlite.db', '')
    opsimName = opsimName.replace('.db', '')
    metadata = args.sqlConstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')

    if not args.skipComp:
        verbose=False
        # Get db connection info, and connect to database.
        dbfile = os.path.join(args.dbDir, args.opsimDb)
        oo = db.OpsimDatabaseV4(dbfile)
        sqlconstraint = args.sqlConstraint
        # Fetch the data from opsim.
        simdata, fields = getData(oo, sqlconstraint)
        # Set up the time bins for the movie slicer.
        start_date = simdata['observationStartMJD'][0]
        if args.movieStepsize == 0:
            bins = simdata['observationStartMJD']
        else:
            end_date = simdata['observationStartMJD'].max()
            bins = np.arange(start_date, end_date+args.movieStepSize/2.0, args.movieStepSize, float)
        if args.addPreviousObs:
            # Go back and grab all the data, including all previous observations.
            if "night =" in sqlconstraint:
                sqlconstraint = sqlconstraint.replace("night =", "night <=")
            elif "night=" in sqlconstraint:
                sqlconstraint = sqlconstraint.replace("night=", "night<=")
            simdata, fields = getData(oo, sqlconstraint)
            # Update the first bin to be prior to the earliest opsim time.
            bins[0] = simdata['observationStartMJD'][0]

        # Run the movie slicer (and at each step, setup opsim slicer and calculate metrics).
        runSlices(opsimName, metadata, simdata, fields, bins, args, oo, verbose=verbose)

    # Need to set up the metrics to get their names, but don't need to have realistic arguments.
    metricList, plotDictList = setupMetrics(opsimName, metadata)
    stitchMovie(metricList, args)
    end_t, start_t = dtime(start_t)
    print('Total time to create movie: ', end_t)
