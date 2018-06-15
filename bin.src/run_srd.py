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
    propids, proptags, sqls, metadata = setSQL(opsdb, sqlConstraint=args.sqlConstraint,
                                               extraMeta=None)

    bdict = {}
    for tag in ['All', 'WFD']:
        fO = batches.fOBatch(colmap=colmap, runName=args.runName,
                             extraSql=sqls[tag], extraMetadata=metadata[tag],
                             ditherStacker=args.ditherStacker)
        bdict.update(fO)
        astrometry = batches.astrometryBatch(colmap=colmap, runName=args.runName,
                                             extraSql=sqls[tag], extraMetadata=metadata[tag],
                                             ditherStacker=args.ditherStacker)
        bdict.update(astrometry)
        rapidrevisit = batches.rapidRevisitBatch(colmap=colmap, runName=args.runName,
                                                 extraSql=sqls[tag], extraMetadata=metadata[tag],
                                                 ditherStacker=args.ditherStacker)
        bdict.update(rapidrevisit)


    return bdict



if __name__ == '__main__':
    args = parseArgs(subdir='srd')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
