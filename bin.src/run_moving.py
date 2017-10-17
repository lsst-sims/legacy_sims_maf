#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np

import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches

# Assumes you have already created observation file,
# This is currently incomplete compared to movingObjects.py! No plotting, no automatic completeness bundles.



def runMetrics(bdict, outDir, resultsDb=None, Hmark=None):
    """
    Run metrics, write basic output in OutDir.

    Parameters
    ----------
    bdict : dict
        The metric bundles to run.
    outDir : str
        The output directory to store results.
    resultsDb : ~lsst.sims.maf.db.ResultsDb, optional
        The results database to use to track metrics and summary statistics.
    Hmark : float, optional
        The Hmark value to add to the completeness bundles plotDicts.

    Returns
    -------
    dict of metricBundles
        The bundles in this dict now contain the metric values as well.
    """
    print("Calculating metric values.")
    bg = mmb.MoMetricBundleGroup(bdict, outDir=outDir, resultsDb=resultsDb)
    # Just calculate here, we'll create the (mostly custom) plots later.
    bg.runAll()





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--obsFile", type=str,
                        help="File containing the observations of the moving objects.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output directory for moving object metrics. Default '.'")
    parser.add_argument("--opsimDb", type=str, default=None,
                        help="Path and filename of opsim db, to write config* files to output directory."
                        " Optional: if not provided, config* files won't be created but analysis will run.")
    parser.add_argument("--hMin", type=float, default=5.0, help="Minimum H value. Default 5.")
    parser.add_argument("--hMax", type=float, default=27.0, help="Maximum H value. Default 27.")
    parser.add_argument("--hStep", type=float, default=0.5, help="Stepsizes in H values.")
    parser.add_argument("--metadata", type=str, default='',
                        help="Base string to add to all metric metadata. Typically the object type.")
    parser.add_argument("--albedo", type=float, default=None,
                        help="Albedo value, to add diameters to upper scales on plots. Default None.")
    parser.add_argument("--hMark", type=float, default=None,
                        help="Add vertical lines at H=hMark on plots. Default None.")
    parser.add_argument("--nYearsMax", type=int, default=10,
                        help="Maximum number of years out to which to evaluate completeness."
                             "Default 10.")
    parser.add_argument("--startTime", type=float, default=59580,
                        help="Time at start of survey (to set time for summary metrics).")
    parser.add_argument("--plotOnly", action='store_true', default=False,
                        help="Reload metric values from disk and replot them.")
    args = parser.parse_args()

    if args.orbitFile is None:
        print('Must specify an orbitFile')
        exit()

    # Default parameters for metric setup.
    stepsize = 365/2.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime
    bins = np.arange(5, 95, 10.)  # binsize to split period (360deg)
    windows = np.arange(1, 200, 15)  # binsize to split time (days)

    if args.plotOnly:
        # Set up resultsDb.
        pass

    else:
        if args.obsFile is None:
            print('Must specify an obsFile when calculating the metrics.')
            exit()
        # Set up resultsDb.
        if not (os.path.isdir(args.outDir)):
            os.makedirs(args.outDir)
        resultsDb = db.ResultsDb(outDir=args.outDir)

        Hrange = np.arange(args.hMin, args.hMax + args.hStep, args.hStep)
        slicer = batches.setupSlicer(args.orbitFile, Hrange, obsFile=args.obsFile)
        opsdb = db.OpsimDatabase(args.opsimDb)
        colmap = batches.getColMap(opsdb)
        bdict = batches.discoveryBatch(slicer, colmap=colmap, runName=args.opsimRun, metadata=args.metadata,
                                       albedo=args.albedo, Hmark=args.hMark, times=times)
        runMetrics(bdict, args.outDir, resultsDb, args.hMark)

    #plotMetrics(allBundles, args.outDir, args.metadata, args.opsimRun, mParams,
    #            Hmark=args.hMark, resultsDb=resultsDb)

    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        utils.writeConfigs(opsdb, args.outDir)
