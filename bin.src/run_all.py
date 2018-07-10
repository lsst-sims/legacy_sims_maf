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

    # Some of these metrics are reproduced in other scripts - srd and cadence
    bdict = {}
    plotbundles = []

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

    # Intranight (pairs/time)
    intranight_all, plots = batches.intraNight(colmap, args.runName, extraSql=args.sqlConstraint,
                                               ditherStacker=args.ditherStacker)
    bdict.update(intranight_all)
    plotbundles.append(plots)
    sql = '(%s) or (%s)' % (sqls['WFD'], sqls['NES'])
    md = 'WFD+' + metadata['NES']
    intranight_wfdnes, plots = batches.intraNight(colmap, args.runName, extraSql=sql,
                                                  extraMetadata=md, ditherStacker=args.ditherStacker)
    bdict.update(intranight_wfdnes)
    plotbundles.append(plots)
    internight_all, plots = batches.interNight(colmap, args.runName, extraSql=args.sqlConstraint,
                                               ditherStacker=args.ditherStacker)
    bdict.update(internight_all)
    plotbundles.append(plots)
    internight_wfd, plots = batches.interNight(colmap, args.runName, extraSql=sqls['WFD'],
                                               extraMetadata=metadata['WFD'],
                                               ditherStacker=args.ditherStacker)
    bdict.update(internight_wfd)
    plotbundles.append(plots)

    # Run all metadata metrics, All and just WFD.
    for tag in ['All', 'WFD']:
        bdict.update(batches.allMetadata(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag],
                                         ditherStacker=args.ditherStacker))

    # Nvisits + m5 maps + Teff maps, All and just WFD.
    for tag in ['All', 'WFD']:
        bdict.update(batches.nvisitsM5Maps(colmap, args.runName, runLength=args.nyears,
                                           extraSql=sqls[tag], extraMetadata=metadata[tag],
                                           ditherStacker=args.ditherStacker))
        bdict.update(batches.tEffMetrics(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag],
                                         ditherStacker=args.ditherStacker))

    # Nvisits per proposal and per night.
    bdict.update(batches.nvisitsPerProp(opsdb, colmap, args.runName,
                                        extraSql=args.sqlConstraint))

    # NVisits alt/az LambertSkyMap (all filters, per filter)
    bdict.update(batches.altazLambert(colmap, args.runName, extraSql=args.sqlConstraint))

    # Slew metrics.
    bdict.update(batches.slewBasics(colmap, args.runName, sqlConstraint=args.sqlConstraint))

    # Open shutter metrics.
    bdict.update(batches.openshutterFractions(colmap, args.runName, extraSql=args.sqlConstraint))

    # Per night and whole survey filter changes.
    bdict.update(batches.filtersPerNight(colmap, args.runName, nights=1, extraSql=args.sqlConstraint))
    bdict.update(batches.filtersWholeSurvey(colmap, args.runName, extraSql=args.sqlConstraint))

    # Hourglass plots.
    bdict.update(batches.hourglassPlots(colmap, args.runName,
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
