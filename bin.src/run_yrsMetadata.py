#!/usr/bin/env python

"""
Run metrics on a single piece of metadata, for years 1, 2, 5, 10
"""

import argparse
import numpy as np
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

    years = np.array([1, 2, 5, 10], dtype=int)
    nights = years * 365.25
    for yr, maxnight in zip(years, nights):
        for tag in tags:
            md = 'year %d' % yr
            sql = 'night <= %f' % maxnight
            if metadata[tag] is not None and len(metadata[tag]) > 0:
                md = metadata[tag] + md
            if sqls[tag] is not None and len(sqls[tag]) > 0:
                sql = '(%s) and (%s)' % (sqls[tag], sql)
            mdMapsDict, mdMapsPlots = batches.metadataMaps(args.metadata, colmap, args.runName,
                                                           extraSql=sql, extraMetadata=md)
            bdict.update(mdMapsDict)
    return bdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run or replot metrics on a single metadata quantity.")
    parser.add_argument("metadata", type=str, help="Name of the metadata quantity to evaluate.")
    parser.add_argument("--noWFD", dest='noWFD', action='store_true', default=False,
                        help="Only run metrics on 'All', not 'WFD'.")
    args = parseArgs(subdir='years', parser=parser)

    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
