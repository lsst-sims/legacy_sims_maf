#!/usr/bin/env python

"""
Run a whole lot of metrics.
"""

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    propids, proptags, sqltags = setSQL(opsdb, sqlConstraint=args.sqlConstraint)

    bdict = {}
    #bdict.update(batches.glanceBatch(colmap=colmap, runName=args.runName, sqlConstraint=args.sqlConstraint))
    bdict.update(batches.intraNight(colmap, args.runName, extraSql=args.sqlConstraint))
    bdict.update(batches.interNight(colmap, args.runName, extraSql=args.sqlConstraint))
    # All metadata, all proposals.
    bdict.update(batches.allMetadata(colmap, args.runName, sqlConstraint=args.sqlConstraint,
                                     metadata='All props'))
    # WFD only.
    bdict.update(batches.allMetadata(colmap, args.runName, sqlConstraint=sqltags['WFD'], metadata='WFD'))
    # number of observations per proposal and per night.
    bdict.update(batches.nvisitsPerProp(opsdb, colmap, args.runName, sqlConstraint=args.sqlConstraint))
    # add nvisits / teff maps.
    # All props
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName, extraSql=args.sqlConstraint))
    # WFD only
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName,
                                       extraSql=sqltags['WFD'], extraMetadata='WFD'))
    # All props.
    bdict.update(batches.tEffMetrics(colmap, args.runName, extraSql=args.sqlConstraint))
    # WFD only.
    bdict.update(batches.tEffMetrics(colmap, args.runName,
                                     extraSql=sqltags['WFD'], extraMetadata='WFD'))
    bdict.update(batches.slewBasics(colmap, args.runName))

    # Whole survey filter changes
    bdict.update(batches.filtersWholeSurveyBatch(colmap, args.runName))

    return bdict



if __name__ == '__main__':
    args = parseArgs(subdir='all')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
