#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambert(colmap, args.runName, extraSql=args.sqlConstraint))

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
