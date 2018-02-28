#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}
    # number of observations per propsosal and per night.
    bdict.update(batches.nvisitsPerProp(opsdb, colmap, args.runName, extraSql=args.sqlConstraint))
    # add nvisits / teff maps.
    propids, proptags, sqls, metadata = setSQL(opsdb, args.sqlConstraint)

    for tag in ['All', 'WFD']:
        bdict.update(batches.nvisitsM5Maps(colmap, args.runName, runLength=args.nyears,
                                           extraSql=sqls[tag], extraMetadata=metadata[tag]))
        bdict.update(batches.tEffMetrics(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag]))

    return bdict


if __name__ == '__main__':
    args = parseArgs('nvisits')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
