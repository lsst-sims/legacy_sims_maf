#!/usr/bin/env python

"""
Run season metrics
"""

from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    # Set up WFD sql constraint
    propids, proptags, sqls, metadata = setSQL(opsdb, sqlConstraint=args.sqlConstraint)

    bdict = {}
    plotbundles = []

    for tag in ['All', 'WFD']:
        season, pb = batches.seasons(colmap=colmap, runName=args.runName,
                                     extraSql=sqls[tag], extraMetadata=metadata[tag],
                                     ditherStacker=args.ditherStacker)
        bdict.update(season)
        plotbundles.append(pb)

    return bdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run or replot metrics on the seasons.")
    args = parseArgs(subdir='all', parser=parser)

    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
