!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches


# Assumes you have run (potentially split) metrics.
# This will join the splits into a single metric output file and add completeness/fraction outputs.
# So - even if you've run run_moving_calc on a single/non-split set of observations, you should still
# run this afterwards.
# It has to know about all of the split output subdirectories.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Join moving object metrics (from splits) for a particular "
                                                 "opsim run. Calculates fractions/completeness also.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--baseDir", type=str, default='.',
                        help="Root directory containing split (or single) metric outputs.")
    parser.add_argument("--split", type=bool, default=True,
                        help="False = single directory of output, in baseDir. Will not join metric outputs."
                             "True = multiple subdirectories [0, 1,..., 9] under baseDir. Will join outputs.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output directory for moving object metrics. Default '.'.")
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

    # Default parameters for completeness metric setup.
    stepsize = 365 / 2.
    times = np.arange(0, args.nYearsMax * 365 + stepsize / 2, stepsize)
    times += args.startTime

    # Outputs from the metrics are generally like so:
    #  <baseDir>/<splitDir>/<metricFileName>
    # - baseDir tends to be <opsimName_orbitRoot> (but is set by user when starting to generate obs.)
    # - splitDir tends to be <orbitRoot_split#> (and is set by observation generation script)
    # - metricFile is <opsimName_metricName_metadata(NEO/L7/etc + metadata from metric script)_MOOB.npz
    #  (the metricFileName is set by the metric generation script - run_moving_calc.py).
    #  (note that split# does not show up in the metricFileName, and is not used in run_moving_calc.py).
    #  ... this lets run_moving_calc.py easily run in parallel on multiple splits.

    # Assume splits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    splits = np.arange(0, 10, 1)
    orbitRoot = args.orbitFile.replace('.txt', '').replace('.des', '').replace('.s3m', '')

    # Scan first splitDir (or baseDir, if not split) for all metric files.
    tempdir = args.baseDir
    if args.split:
        tempdir = os.path.join(args.baseDir, orbitRoot + "_" + splits[0])
    metricfiles = glob.glob(tempdir + '*MOOB.npz')
    # Identify metric names that we want to join - cannot join parent Discovery*Metric data.
    metricNames = []
    for m in metricfiles:
        # Pull out metric name only.
        mname = '_'.join(m.rstrip('_MOOB.npz').split('/')[-1].split('_')[1:])
        # Hack out raw Discovery outputs.
        if mname.startswith('Discovery_'):
            if mname.startswith('Discovery_Time'):
                metricNames.append(mname)
            elif mname.startswith('Discovery_N_Chances'):
                metricNames.append(mname)
            else:
                pass
        else:
            metricNames.append(mname)

    if len(metricNames) == 0:
        print(f"Could not read any metric files from {tempdir}")
        exit()

    # Generate the column map.
    if args.opsimDb is not None:
        opsdb = db.OpsimDatabase(args.opsimDb)
        colmap = batches.getColMap(opsdb)
        opsdb.close()
    else:
        # Use the default (currently, v4).
        colmap = batches.ColMapDict()

    # Set up resultsDb, to save fraction/completeness values.
    if not (os.path.isdir(args.outDir)):
        os.makedirs(args.outDir)
    resultsDb = db.ResultsDb(outDir=args.outDir)

    # Read and combine the metric files.
    readAndCombine(orbitRoot, baseDir, splits, metricfile + "_MOOB.npz")