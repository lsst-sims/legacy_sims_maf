#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches

"""Calculate metric values for an input population. Can be used on either a split or complete population.
If running on a split population for later re-combining, use the complete set of orbits as the 
'orbitFile'. Assumes you have already created the moving object observation files.
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--characterization", type=str, help="Inner/Outer solar system characterization?")
    parser.add_argument("--obsFile", type=str,
                        help="File containing the observations of the moving objects.")
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

    if args.characterization.lower() not in ['inner', 'outer']:
        print('Please choose either inner (asteroid) or outer (TNO/SDO?) for characterization.')
        exit()

    # Default parameters for metric setup.
    stepsize = 365/2.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime

    if args.obsFile is None:
        print('Must specify an obsFile when calculating the metrics.')
        exit()
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

    slicer = batches.setupMoSlicer(args.orbitFile, Hrange, obsFile=args.obsFile)
    # Run discovery metrics using 'trailing' losses
    bdictT, pbundleT = batches.quickDiscoveryBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                                 metadata=args.metadata, detectionLosses='trailing',
                                                 albedo=args.albedo, Hmark=args.hMark)
    # Run these discovery metrics
    print("Calculating quick discovery metrics with simple trailing losses.")
    bg = mmb.MoMetricBundleGroup(bdictT, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all discovery metrics using 'detection' losses
    bdictD, pbundleD = batches.quickDiscoveryBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                                 metadata=args.metadata, detectionLosses='detection',
                                                 albedo=args.albedo, Hmark=args.hMark)
    bdict, pbundle = batches.discoveryBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                            metadata=args.metadata,
                                            detectionLosses='detection', albedo=args.albedo,
                                            Hmark=args.hMark)
    bdictD.update(bdict)
    pbundleD.append(pbundle)

    # Run these discovery metrics
    print("Calculating full discovery metrics with detection losses.")
    bg = mmb.MoMetricBundleGroup(bdictD, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    # Run all characterization metrics
    if args.characterization.lower() == 'inner':
        bdictC, pbundle = batches.characterizationInnerBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                                             metadata=args.metadata, albedo=args.albedo,
                                                             Hmark=args.hMark, constraint=None)
    elif args.characterization.lower() == 'outer':
        bdictC, pbundle = batches.characterizationOuterBatch(slicer, colmap=colmap, runName=args.opsimRun,
                                                             metadata=args.metadata, albedo=args.albedo,
                                                             Hmark=args.hMark, constraint=None)
    # Run these characterization metrics
    print("Calculating characterization metrics.")
    bg = mmb.MoMetricBundleGroup(bdictC, outDir=args.outDir, resultsDb=resultsDb)
    bg.runAll()

    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        utils.writeConfigs(opsdb, args.outDir)
        opsdb.close()
