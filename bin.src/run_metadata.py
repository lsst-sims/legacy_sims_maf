#!/usr/bin/env python

"""
Run a whole lot of metrics.
"""

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    # Set up WFD sql constraint, if possible.
    propids, proptags, sqls, metadata = setSQL(opsdb, sqlConstraint=args.sqlConstraint)
    if 'WFD' in sqls:
        tags = ['All', 'WFD']
    else:
        tags = ['All']
    # Set up appropriate metadata - need to combine args.sqlConstraint

    # Some of these metrics are reproduced in other scripts - srd and cadence
    bdict = {}
    plotbundles = []

    for tag in tags:
        fO = batches.fOBatch(colmap=colmap, runName=args.runName,
                             extraSql=sqls[tag], extraMetadata=metadata[tag])
        bdict.update(fO)
        astrometry = batches.astrometryBatch(colmap=colmap, runName=args.runName,
                                             extraSql=sqls[tag], extraMetadata=metadata[tag])
        bdict.update(astrometry)
        rapidrevisit = batches.rapidRevisitBatch(colmap=colmap, runName=args.runName,
                                                 extraSql=sqls[tag], extraMetadata=metadata[tag])
        bdict.update(rapidrevisit)

    # Intranight (pairs/time)
    intranight_all, plots = batches.intraNight(colmap, args.runName, extraSql=args.sqlConstraint)
    bdict.update(intranight_all)
    plotbundles.append(plots)
    if ('WFD' in sqls) and ('NES' in sqls):
        sql = '(%s) or (%s)' % (sqls['WFD'], sqls['NES'])
        md = 'WFD+' + metadata['NES']
        intranight_wfdnes, plots = batches.intraNight(colmap, args.runName, extraSql=sql,
                                                      extraMetadata=md)
        bdict.update(intranight_wfdnes)
        plotbundles.append(plots)

    # Internight (nights between visits)
    for tag in tags:
        internight, plots = batches.interNight(colmap, args.runName, extraSql=sqls[tag],
                                               extraMetadata=metadata[tag])
        bdict.update(internight)
        plotbundles.append(plots)

    # Intraseason (length of season)
    for tag in tags:
        season, plots = batches.seasons(colmap=colmap, runName=args.runName,
                                        extraSql=sqls[tag], extraMetadata=metadata[tag])
        bdict.update(season)
        plotbundles.append(plots)

    # Run all metadata metrics, All and just WFD.
    for tag in tags:
        bdict.update(batches.allMetadata(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag]))

    # Nvisits + m5 maps + Teff maps, All and just WFD.
    for tag in tags:
        bdict.update(batches.nvisitsM5Maps(colmap, args.runName, runLength=args.nyears,
                                           extraSql=sqls[tag], extraMetadata=metadata[tag]))
        bdict.update(batches.tEffMetrics(colmap, args.runName, extraSql=sqls[tag],
                                         extraMetadata=metadata[tag]))

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
    args = parseArgs(subdir='metadata')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
