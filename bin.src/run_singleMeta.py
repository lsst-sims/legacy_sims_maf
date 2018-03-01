#!/usr/bin/env python

"""
Run metrics on a single piece of metadata.
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
    # Set up appropriate metadata - need to combine args.sqlConstraint

    bdict = {}

    if args.noWFD:
        tags = ['All']
    else:
        tags = ['All', 'WFD']
    # Single metadata, All and WFD.
    for tag in tags:
        bdict.update(batches.metadataBasics(args.metadata, colmap, args.runName, extraSql=sqls[tag],
                                            extraMetadata=metadata[tag]))
    return bdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run or replot metrics on a single metadata quantity.")
    parser.add_argument("metadata", type=str, help="Name of the metadata quantity to evaluate.")
    parser.add_argument("--noWFD", dest='noWFD', action='store_true', default=False,
                        help="Only run metrics on 'All', not 'WFD'.")
    args = parseArgs(subdir='single', parser=parser)

    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
