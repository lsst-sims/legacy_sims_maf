#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.batches as batches

"""Calculate completeness and fractions for moving object metrics."""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--workDir", type=str, default='.',
                        help="Output (and input) directory for moving object metrics. Default '.'.")
    parser.add_argument('--metadata', type=str, default=None,
                        help="Select only files matching this metadata string. Default None (all files).")
    parser.add_argument("--hMark", type=float, default=None,
                        help="H value at which to calculate cumulative/differential completeness, etc."
                             "Default (None) will be set to plotDict value or median of H range.")
    parser.add_argument("--nYearsMax", type=int, default=10,
                        help="Maximum number of years out to which to evaluate completeness."
                             "Default 10.")
    parser.add_argument("--startTime", type=float, default=59853,
                        help="Time at start of survey (to set time for summary metrics).")
    args = parser.parse_args()

    # Default parameters for metric setup.
    stepsize = 365/6.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime

    # Create a results Db.
    resultsDb = db.ResultsDb(outDir=args.workDir)

    # Just read in all metrics in the (joint or single) directory, then run completeness and fraction
    # summaries, using the methods in the batches.
    if args.metadata is None:
        matchstring = os.path.join(args.workDir, '*MOOB.npz')
    else:
        matchstring = os.path.join(args.workDir, f'*{args.metadata}*MOOB.npz')
    metricfiles = glob.glob(matchstring)
    metricNames = []
    for m in metricfiles:
        mname = os.path.split(m)[-1].replace('_MOOB.npz', '')
        metricNames.append(mname)

    bdict = {}
    for mName, mFile in zip(metricNames, metricfiles):
        bdict[mName] = mmb.createEmptyMoMetricBundle()
        bdict[mName].read(mFile)

    first = bdict[metricNames[0]]
    figroot = f'{first.runName}'
    if args.metadata != '.':
        figroot += f'_{args.metadata}'

    # Calculate completeness. This utility writes these to disk.
    bdictCompleteness = batches.runCompletenessSummary(bdict, args.hMark, times, args.workDir, resultsDb)

    # Plot some of the completeness results.
    batches.plotCompleteness(bdictCompleteness, figroot=figroot,
                             resultsDb=resultsDb, outDir=args.workDir)

    # Calculate fractions of population for characterization. This utility writes these to disk.
    bdictFractions = batches.runFractionSummary(bdict, args.hMark, args.workDir, resultsDb)
    # Plot the fractions for colors and lightcurves.
    batches.plotFractions(bdictFractions, figroot=figroot,
                          resultsDb=resultsDb, outDir=args.workDir)

    # Plot nObs and arcLength.
    for k in bdict:
        if 'NObs' in k:
            batches.plotSingle(bdict[k], resultsDb=resultsDb, outDir=args.workDir)
        if 'ObsArc' in k:
            batches.plotSingle(bdict[k], resultsDb=resultsDb, outDir=args.workDir)

    # Plot likelihood of detecting activity.
    batches.plotActivity(bdict, figroot=figroot, resultsDb=resultsDb, outDir=args.workDir)
