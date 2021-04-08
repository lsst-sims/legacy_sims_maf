#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}
    bdict.update(batches.hourglassPlots(colmap, args.runName,
                                        nyears=args.nyears, extraSql=args.sqlConstraint))
    return bdict


if __name__ == '__main__':
    args = parseArgs(subdir='hourglass')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        raise ValueError('Cannot replot hourglass metric data, as it is not saved to disk.')
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
