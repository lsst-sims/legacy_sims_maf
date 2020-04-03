#!/usr/bin/env python

"""
Run the SRD metrics.
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = batches.scienceRadarBatch(colmap=None, runName=args.runName,
                                      extraSql=args.sqlConstraint, extraMetadata=None,
                                      nside=64, benchmarkArea=18000, benchmarkNvisits=825, DDF=True)
    return bdict

if __name__ == '__main__':
    args = parseArgs(subdir='science')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
