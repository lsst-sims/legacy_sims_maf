#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}
    bdict.update(batches.intraNight(colmap, args.runName))
    return bdict


if __name__ == '__main__':
    args = parseArgs('cadence')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()

