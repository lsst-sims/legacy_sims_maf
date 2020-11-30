#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches

"""Calculate only the 'quickDiscoveryBatch' metrics for an input population. 
Can be used on either a split or complete population, assumes you have already created the observation files.
Use the complete set of orbits as the 'orbitFile'.
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--obsFile", type=str,
                        help="File(s) containing the observations of the moving objects."
                             "If providing a list, surround filenames with quotes.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output directory for moving object metrics. Default '.'.")
    parser.add_argument("--opsimDb", type=str, default=None,
                        help="Path and filename of opsim db, to write config* files to output directory."
                        " Optional: if not provided, config* files won't be created but analysis will run.")
    parser.add_argument("--hMin", type=float, default=5.0, help="Minimum H value. Default 5.")
    parser.add_argument("--hMax", type=float, default=27.0, help="Maximum H value. Default 27.")
    parser.add_argument("--hStep", type=float, default=0.2, help="Stepsizes in H values.")
    parser.add_argument("--metadata", type=str, default='',
                        help="Base string to add to all metric metadata. Typically the object type.")
    parser.add_argument("--albedo", type=float, default=None,
                        help="Albedo value, to add diameters to upper scales on plots. Default None.")
    parser.add_argument("--hMark", type=float, default=None,
                        help="Add vertical lines at H=hMark on plots. Default None.")
    parser.add_argument("--nYearsMax", type=int, default=10,
                        help="Maximum number of years out to which to evaluate completeness."
                             "Default 10.")
    parser.add_argument("--startTime", type=float, default=59853,
                        help="Time at start of survey (to set time for summary metrics).")
    args = parser.parse_args()

    if args.orbitFile is None:
        print('Must specify an orbitFile')
        exit()

    # Default parameters for metric setup.
    stepsize = 365/2.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime

    # Modified to handle either a single observation file or multiple files.
    # This script will run all of the individual calculations serially, then join the results,
    # and calculate completeness.
    if args.obsFile is None:
        print('Must specify an obsFile when calculating the metrics.')
        exit()
    elif isinstance(args.obsFile, list):
        obsFiles = args.obsFile
    else:
        obsFiles = [args.obsFile]
    print(f'Will loop through {len(obsFiles)} observation files')
    # Set up resultsDb.
    if not (os.path.isdir(args.outDir)):
        os.makedirs(args.outDir)
    resultsDb = db.ResultsDb(outDir=args.outDir)

    Hrange = np.arange(args.hMin, args.hMax + args.hStep, args.hStep)
    if args.hMark is None:
        hIdx = int(len(Hrange)/2)
        args.hMark = Hrange[hIdx]

    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        colmap = batches.getColMap(opsdb)
        opsdb.close()
    else:
        # Use the default (currently, v4).
        colmap = batches.ColMapDict()


    # Loop through calculation of metrics
    tempRoot = 'quick_subset'
    for i, obsFile in enumerate(obsFiles):
        slicer = batches.setupMoSlicer(args.orbitFile, Hrange, obsFile=obsFile)
        # Run discovery metrics using 'trailing' losses
        bdictD, pbundleD = batches.quickDiscoveryBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                                     metadata=args.metadata, detectionLosses='detection',
                                                     albedo=args.albedo, Hmark=args.hMark)
        # Run these discovery metrics - write to subdirectories (because their names will conflict)
        print("Calculating quick discovery metrics with basic detection losses.")
        bg = mmb.MoMetricBundleGroup(bdictD, outDir=os.path.join(args.outDir, f"{tempRoot}_{i}"),
                                     resultsDb=None)
        bg.runAll()

    # Join metric results together - write into outDir
    # Scan first subset output directory for metric files
    tempdir = os.path.join(args.outDir, f'{tempRoot}_0')
    print(f'# Joining files from {tempRoot}_[0, {len(obsFiles)-1}]; will use {tempdir} to find metric names.')

    metricfiles = glob.glob(os.path.join(tempdir, '*Discovery*MOOB.npz'))
    # Identify metric names that we want to join.
    metricNames = []
    for m in metricfiles:
        mname = os.path.split(m)[-1]
        # Hack out raw Discovery outputs. We don't want to join the raw discovery files.
        # This is a hack because currently we're just pulling out _Time and _N_Chances to join.
        if 'Discovery' in mname:
            if 'Discovery_Time' in mname:
                metricNames.append(mname)
            elif 'Discovery_N_Chances' in mname:
                metricNames.append(mname)
            else:
                pass
        else:
            metricNames.append(mname)

    # Read and combine the metric files.
    splits = np.arange(0, len(obsFiles), 1)
    for m in metricNames:
        b = batches.readAndCombine(tempRoot, args.outDir, splits, m)
        b.write(outDir=args.outDir)

    # Calculate fractions/completeness
    # Create a results Db.
    resultsDb = db.ResultsDb(outDir=args.outDir)

    # Just read in all metrics in the (joint or single) directory, then run completeness and fraction
    # summaries, using the methods in the batches.
    if args.metadata is None:
        matchstring = os.path.join(args.outDir, '*MOOB.npz')
    else:
        matchstring = os.path.join(args.outDir, f'*{args.metadata}*MOOB.npz')
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
    if args.metadata is not '.':
        figroot += f'_{args.metadata}'

    # Calculate completeness. This utility writes these to disk.
    bdictCompleteness = batches.runCompletenessSummary(bdict, args.hMark, times, args.outDir, resultsDb)

    # Plot some of the completeness results.
    batches.plotCompleteness(bdictCompleteness, figroot=figroot,
                             resultsDb=resultsDb, outDir=args.outDir)


    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        utils.writeConfigs(opsdb, args.outDir)
        opsdb.close()
