import matplotlib
matplotlib.use("Agg")
import os
import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import warnings
import fnmatch
import shutil
import unittest


def setupMovieSlicer(simdata, binsize = 365.25, cumulative=True):
    ms = slicers.MovieSlicer(sliceColName='expMJD', binsize=binsize, cumulative=cumulative)
    ms.setupSlicer(simdata)
    return ms

def setupHealpixSlicer(simdatasubset, racol, deccol, nside):
    hs = slicers.HealpixSlicer(nside=nside, spatialkey1=racol, spatialkey2=deccol, plotFuncs='plotSkyMap')
    hs.setupSlicer(simdatasubset)
    return hs


def setupMetrics():
    # Set up metrics.
    metricList = []
    metricList.append(metrics.CountMetric('expMJD', metricName='Number of Visits',
                                          plotDict={'logScale':False,
                                                      'colorMin':0, 'colorMax':300,
                                                      'cbarFormat': '%d'}))
    return metricList

def run(opsimName, metadata, simdata, metricList, largs, ffmpeg = True):
    """Do the work to run the movie slicer, and at each step, setup the healpix slicer and run the metrics,
    making the plots."""
    sliceformat = '%04d'
    #calculation of metrics
    if largs.skipComp:
        # Set up movie slicer
        movieslicer = setupMovieSlicer(simdata, binsize = largs.movieStepsize, cumulative=largs.cumulative)
        start_date = movieslicer[0]['slicePoint']['binLeft']
        # Run through the movie slicer slicePoints:
        for i, movieslice in enumerate(movieslicer):
            timeInterval = '%.2f to %.2f' %(movieslice['slicePoint']['binLeft']-
                                            start_date, movieslice['slicePoint']['binRight']-start_date)
            slicenumber = '%.4d' %(i)
            #adding day number to title of plot
            if largs.cumulative:
                for metric in metricList:
                    metric.plotDict['label'] = 'day: ' + str(i*largs.movieStepsize) + '\n' + largs.sqlConstraint
            else:
                for metric in metricList:
                    metric.plotDict['label'] = 'time interval: ' + '\n' + timeInterval + '\n'+ '\n' + largs.sqlConstraint

            # Identify the subset of simdata in the movieslicer 'data slice'
            simdatasubset = simdata[movieslice['idxs']]
            # Set up healpix slicer on subset of simdata provided by movieslicer
            hs = setupHealpixSlicer(simdatasubset, 'ditheredRA', 'ditheredDec', largs.nside)
            # Set up sliceMetric to handle healpix slicer + metrics calculation + plotting
            sm = sliceMetrics.RunSliceMetric(outDir=largs.outDir, useResultsDb=False,
                                            figformat='png', dpi=72, thumbnail=False)
            sm.setSlicer(hs)
            sm.setMetrics(metricList)
            # Calculate metric data values for simdatasubset
            sm.runSlices(simdatasubset, simDataName=opsimName)
            # Plot data for this slice of the movie, adding slicenumber as a suffix for output plots
            sm.plotAll(outfileSuffix=slicenumber, closefig=True)
            # Write the data -- uncomment if you want to do this.
            # sm.writeAll(outfileSuffix=slicenumber)

    else:
        #create movieslicer to call plotMovie
        movieslicer = slicers.MovieSlicer()

    outfileroots = []
    for metric in metricList:
        #Create plot names
        Mname = metric.name.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        dbName = largs.opsimDb.strip('_sqlite.db')
        outfileroots.append(dbName + '_' + Mname + '_' + 'HEAL')

    #if a movieLength was specified
    if largs.movieLength != 0.0:
        if largs.skipComp:
            largs.ips = len(movieslicer)/largs.movieLength
        else:
            #figure out how many images there are.
            n_images = len(fnmatch.filter(os.listdir(largs.outDir), outfileroots[0] + '*SkyMap.png'))
            print outfileroots[0]
            if n_images == 0:
                warnings.warn('Targeted files did not match directory contents. Make sure the parameters of this run, match the files. (for instance metrics)')
            #calculate images/second rate
            largs.ips = n_images/largs.movieLength
        print "for a movie length of " + str(largs.movieLength) + " IPS set to: ", largs.ips

    if largs.fps == 0.0:
            warnings.warn('(FPS of 0.0) Setting fps equal to ips, up to a value of 30fps.')
            if largs.ips <= 30.0:
                largs.fps = largs.ips
            else:
                largs.fps = 30.0
    # Create the movie for every metric that was run
    if ffmpeg:
        for outroot in outfileroots:
            out = outroot.split('/')[-1]
            movieslicer.plotMovie(out, sliceformat, plotType='SkyMap', figformat='png',
                                  outDir=largs.outDir, ips=largs.ips, fps=largs.fps)


class TestMovieSlicer(unittest.TestCase):
    """Run the movie slicer.  Check that it outputs mp4 and gif files. """

    class setArgs(object):
        def __init__(self):
            self.filepath = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/')
            self.opsimDb = 'sqlite:///' + self.filepath + 'opsimblitz1_1133_sqlite.db'
            self.skipComp = False
            self.movieStepsize = 5
            self.cumulative=True
            self.sqlConstraint = 'night < 10'
            self.nside = 16
            self.outDir = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/movieTest')
            self.cumulative = True
            self.ips = 10.
            self.fps = 0.0
            self.skipComp = True
            self.movieLength = 0.0

    def setUp(self):
        self.args = self.setArgs()
        os.mkdir(self.args.outDir)
        # Assume buildbot doesn't have ffmpeg.
        ffPath = None #distutils.spawn.find_executable('ffmpeg')
        if ffPath is None:
            self.ffmpeg = False
        else:
            self.ffmpeg = True

    def testRun(self):
        oo = db.OpsimDatabase(self.args.opsimDb)
        opsimName = oo.fetchOpsimRunName()
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
        simdata = oo.fetchMetricData(colnames, self.args.sqlConstraint)
        # Run the movie slicer (and at each step, healpix slicer and calculate metrics).
        comment = self.args.sqlConstraint.replace('=','').replace('filter','').replace("'",'').replace('"','').replace('/','.')
        ##for goodness sake, why not just pass it args itself?
        gm = run(opsimName, comment, simdata, metricList, self.args, ffmpeg = self.ffmpeg)
        mapsOut = ['opsimblitz1_1133_Number_of_Visits_HEAL_0000_SkyMap.png',
                   'opsimblitz1_1133_Number_of_Visits_HEAL_0001_SkyMap.png']
        for filename in mapsOut:
            assert(os.path.isfile(os.path.join(self.args.outDir,filename)))

        if self.ffmpeg:
            moviesOut = ['opsimblitz1_1133_Number_of_Visits_HEAL_SkyMap_10.0_10.0.gif',
                         'opsimblitz1_1133_Number_of_Visits_HEAL_SkyMap_10.0_10.0.mp4']
            for filename in moviesOut:
                assert(os.path.isfile(os.path.join(self.args.outDir,filename)))


    def tearDown(self):
        direc = os.path.join(os.getenv('SIMS_MAF_DIR'), 'tests/movieTest')
        if os.path.isdir(direc):
            shutil.rmtree(direc)


if __name__ == "__main__":
    unittest.main()
