#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

import lsst.sims.maf.batches as batches


# Assumes you have run (potentially split) metrics.
# This will join the splits into a single metric output file.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Join moving object metrics (from splits) for a particular "
                                                 "opsim run.")
    parser.add_argument("--orbitFile", type=str, help="File containing the moving object orbits.")
    parser.add_argument("--opsimRun", type=str, default='opsim',
                        help="Name of opsim run. Default 'opsim'.")
    parser.add_argument("--baseDir", type=str, default='.',
                        help="Root directory containing split (or single) metric outputs.")
    parser.add_argument("--outDir", type=str, default=None,
                        help="Output directory for moving object metrics. Default [orbitRoot]")
    args = parser.parse_args()


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

    if args.outDir is not None:
        outDir = args.outDir
    else:
        outDir = f'{orbitRoot}'

    # Scan first splitDir for all metric files.
    tempdir = os.path.join(args.baseDir, orbitRoot + "_" + splits[0])
    metricfiles = glob.glob(tempdir + '*MOOB.npz')
    # Identify metric names that we want to join - cannot join parent Discovery*Metric data.
    metricNames = []
    for m in metricfiles:
        # Pull out metric name only.
        mname = '_'.join(m.rstrip('_MOOB.npz').split('/')[-1].split('_')[1:])
        # Hack out raw Discovery outputs. We don't want to join the raw discovery files.
        # This is a hack because currently we're just pulling out _Time and _N_Chances to join.
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

    # Read and combine the metric files.
    for m in metricNames:
        b = batches.readAndCombine(orbitRoot, baseDir, splits, m + "_MOOB.npz")
        b.write(outDir=outDir)
