#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}

    b, p = batches.meanRADec(colmap, args.runName, extraSql=args.sqlConstraint)
    bdict.update(b)

    b, p = batches.eastWestBias(colmap, args.runName, extraSql=args.sqlConstraint)
    bdict.update(b)

    return bdict


if __name__ == '__main__':
    args = parseArgs(subdir = 'skycoverage')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
