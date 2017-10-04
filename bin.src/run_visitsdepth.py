#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *

def setBatches(opsdb, colmap, args):
    propids, proptags, sqltags = setSQL(opsdb)
    bdict = {}
    # All props
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName))
    # WFD only
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName,
                                       extraSql=sqltags['WFD'], extraMetadata='WFD'))
    # All props.
    bdict.update(batches.tEffMetrics(colmap, args.runName))
    # WFD only.
    bdict.update(batches.tEffMetrics(colmap, args.runName,
                                     extraSql=sqltags['WFD'], extraMetadata='WFD'))
    return bdict


if __name__ == '__main__':
    args = parseArgs('nvisits')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    print('Set up %d metric bundles.' % (len(bdict)))
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()