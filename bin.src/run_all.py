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
    # Set up WFD sql constraint
    propids, proptags, sqls, metadata = setSQL(opsdb, sqlConstraint=args.sqlConstraint)
    # Set up appropriate metadata - need to combine args.sqlConstraint


    bdict = {}

    # These metrics are reproduced in other scripts - srd and cadence
    """
    for tag in ['All', 'WFD']:
        sql = sqls[tag]
        md = metadata[tag]
        fO = batches.fOBatch(colmap=colmap, runName=args.runName,
                             extraSql=sql, extraMetadata=md)
        bdict.update(fO)
        astrometry = batches.astrometryBatch(colmap=colmap, runName=args.runName,
                                             extraSql=sql, extraMetadata=md)
        bdict.update(astrometry)
        rapidrevisit = batches.rapidRevisitBatch(colmap=colmap, runName=args.runName,
                                                 extraSql=sql, extraMetadata=md)
        bdict.update(rapidrevisit)
    bdict.update(batches.glanceBatch(colmap=colmap, runName=args.runName, sqlConstraint=args.sqlConstraint))
    bdict.update(batches.intraNight(colmap, args.runName, extraSql=args.sqlConstraint))
    bdict.update(batches.interNight(colmap, args.runName, extraSql=args.sqlConstraint))
    """

    # All metadata, All and WFD.
    for tag in ['All', 'WFD']:
        bdict.update(batches.allMetadata(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag]))

    # number of observations per proposal and per night.
    bdict.update(batches.nvisitsPerProp(opsdb, colmap, args.runName,
                                        extraSql=args.sqlConstraint))

    # Nvisits + coadd depths maps, Teff maps.
    for tag in ['All', 'WFD']:
        bdict.update(batches.nvisitsM5Maps(colmap, args.runName, runLength=args.nyears,
                                           extraSql=sqls[tag], extraMetadata=metadata[tag]))
        bdict.update(batches.tEffMetrics(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag]))

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambBatch(colmap, args.runName, extraSql=args.sqlConstraint))

    # Slew metrics.
    bdict.update(batches.slewBasics(colmap, args.runName, sqlConstraint=args.sqlConstraint))

    # Per night and whole survey filter changes
    bdict.update(batches.filtersPerNightBatch(colmap, args.runName, nights=1, extraSql=args.sqlConstraint))
    bdict.update(batches.filtersWholeSurveyBatch(colmap, args.runName, extraSql=args.sqlConstraint))

    # Hourglass plots
    bdict.update(batches.hourglassBatch(colmap, args.runName,
                                        nyears=args.nyears, extraSql=args.sqlConstraint))

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
