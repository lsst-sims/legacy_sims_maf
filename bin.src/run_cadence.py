#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from .run_generic import *

def setBatches(opsdb, colmap, runName):
    bdict = {}
    bdict.update(batches.intraNight(colmap, runName))
    return bdict



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the survey at a glance bundle.")
    parser.add_argument("dbfile", type=str, help="Sqlite file of observations (full path).")
    parser.add_argument("--runName", type=str, default=None, help="Run name."
                                                                  "Default is based on dbfile name.")
    parser.add_argument("--outDir", type=str, default=None, help="Output directory."
                                                                 "Default is runName/cadence.")
    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values from disk and re-plot them.")
    args = parser.parse_args()

    if args.runName is None:
        runName = os.path.basename(args.dbFile).replace('_sqlite.db', '')
        runName = runName.replace('.db', '')
    else:
        runName = args.runName

    if args.outDir is None:
        outDir = os.path.join(runName, "cadence")
    else:
        outDir = args.outDir

    opsdb, colmap = connectDb(args.dbfile)
    if args.plotOnly:
        replotBatch(opsdb, colmap, outDir=outDir, runName=runName)
    else:
        runBatch(opsdb, colmap, outDir=outDir, runName=runName)

    opsdb.close()
