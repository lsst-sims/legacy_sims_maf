#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}
    bdict.update(batches.glanceBatch(colmap, args.runName, sqlConstraint=args.sqlConstraint))
    bdict.update(batches.fOBatch(colmap, args.runName, extraSql=args.sqlConstraint))
    return bdict


if __name__ == '__main__':
    args = parseArgs(subdir='glance')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
