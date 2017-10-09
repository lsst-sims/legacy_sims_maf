#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.utils as mafUtils
from run_generic import parseArgs, connectDb


def runSlew(opsdb, colmap, args):
    resultsDb = db.ResultsDb(outDir=args.outDir)
    bdict = batches.slewBasics(colmap, args.runName)
    dbTable = None
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, dbTable=dbTable)
    group.runAll()
    group.plotAll()
    if 'slewStatesTable' in colmap:
        bdict = batches.slewAngles(colmap, args.runName)
        dbTable = colmap['slewStatesTable']
        group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, dbTable=dbTable)
        group.runAll()
        group.plotAll()
    if 'slewSpeedsTable' in colmap:
        bdict = batches.slewSpeeds(colmap, args.runName)
        dbTable = colmap['slewSpeedsTable']
        group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, dbTable=dbTable)
        group.runAll()
        group.plotAll()
    if 'slewActivitiesTable' in colmap:
        nslews = opsdb.fetchTotalSlewN()
        bdict = batches.slewActivities(colmap, args.runName, totalSlewN=nslews)
        dbTable = colmap['slewActivitiesTable']
        group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, dbTable=dbTable)
        group.runAll()
        group.plotAll()
    resultsDb.close()
    mafUtils.writeConfigs(opsdb, args.outDir)


def replotSlew(opsdb, colmap, args):
    print('Only replots slew basics batch.')
    bdict = batches.slewBasics(colmap, args.runName)
    resultsDb = db.ResultsDb(outDir=args.outDir)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
    group.readAll()
    group.plotAll()
    resultsDb.close()


if __name__ == '__main__':
    args = parseArgs('slew')
    opsdb, colmap = connectDb(args.dbfile)
    if args.plotOnly:
        replotSlew(opsdb, colmap, args)
    else:
        runSlew(opsdb, colmap, args)
    opsdb.close()

