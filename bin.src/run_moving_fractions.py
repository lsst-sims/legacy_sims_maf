#!/usr/bin/env python

import os
import argparse
import numpy as np

import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mmb
import lsst.sims.maf.utils as utils
import lsst.sims.maf.batches as batches

"""Calculate completeness and fractions for moving object metrics."""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run moving object metrics for a particular opsim run.")
    parser.add_argument("--outDir", type=str, default='.',
                        help="Output (and input) directory for moving object metrics. Default '.'.")
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

    # hstep/hmin/hmax are defined when the original H range is defined and the metric run.
    # However, hMark can be changed here for summary stats.

    # Default parameters for metric setup.
    stepsize = 365/2.
    times = np.arange(0, args.nYearsMax*365 + stepsize/2, stepsize)
    times += args.startTime

    # Create a results Db.
    resultsDb = db.ResultsDb(outDir=args.outDir)

    # Just read in all metrics in the (joint or single) directory, then run completeness and fraction
    # summaries, using the methods in the batches.



