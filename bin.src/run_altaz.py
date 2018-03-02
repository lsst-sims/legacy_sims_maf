#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}

    # NVisits alt/az HealpixSkyMap (all filters, per filter)
    bdict.update(batches.altazHealBatch(colmap, args.runName, extraSql=args.sqlConstraint))

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambBatch(colmap, args.runName, extraSql=args.sqlConstraint))

    return bdict


if __name__ == '__main__':
    args = parseArgs(subdir = 'altaz')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
