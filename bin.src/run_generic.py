#!/usr/bin/env python

"""
This is the basis for other scripts to run specialized sets of batches.
See run_glance and run_cadence for examples.
"""

from __future__ import print_function

import os
import argparse
import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as mafUtils
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.batches as batches

"""
def setBatches(opsdb, colmap, args):
    propids, proptags, sqltags = setSQL(opsdb)

    bdict = {}
    bdict.update(batches.glanceBatch(colmap, runName))
    bdict.update(batches.intraNight(colmap, runName))
    # All metadata, all proposals.
    bdict.update(batches.allMetadata(colmap, runName, sqlconstraint='', metadata='All props'))
    # WFD only.
    bdict.update(batches.allMetadata(colmap, runName, sqlconstraint=sqltags['WFD'], metadata='WFD'))
    return bdict
"""

def connectDb(dbfile):
    version = db.testOpsimVersion(dbfile)
    if version == "Unknown":
        opsdb = db.Database(dbfile)
        colmap = batches.ColMapDict('barebones')
    elif version == "V3":
        opsdb = db.OpsimDatabaseV3(dbfile)
        colmap = batches.ColMapDict('OpsimV3')
    elif version == "V4":
        opsdb = db.OpsimDatabaseV4(dbfile)
        colmap = batches.ColMapDict('OpsimV4')
    return opsdb, colmap


def setSQL(opsdb):
    # Fetch the proposal ID values from the database
    propids, proptags = opsdb.fetchPropInfo()
    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
    wfdWhere = opsdb.createSQLWhere('WFD', proptags)
    ddWhere = opsdb.createSQLWhere('DD', proptags)
    sqltags = {'WFD': wfdWhere, 'DD': ddWhere}
    return (propids, proptags, sqltags)


def run(bdict, opsdb, colmap, args):
    resultsDb = db.ResultsDb(outDir=args.outDir)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
    group.runAll()
    group.plotAll()
    resultsDb.close()
    mafUtils.writeConfigs(opsdb, args.outDir)


def replot(bdict, opsdb, colmap, args):
    resultsDb = db.ResultsDb(outDir=args.outDir)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb)
    group.readAll()
    group.plotAll()
    resultsDb.close()


def parseArgs(subdir='out', parser=None):
    if parser is None:
        # Let the user set up their own argparse Parser, in case they need to add new args.
        parser = argparse.ArgumentParser(description="Run or replot a set of metric bundles.")
    # Things we always need.
    parser.add_argument("dbfile", type=str, help="Sqlite file of observations (full path).")
    parser.add_argument("--runName", type=str, default=None, help="Run name."
                                                                  "Default is based on dbfile name.")
    parser.add_argument("--outDir", type=str, default=None, help="Output directory."
                                                                 "Default is runName/%s." % (subdir))
    parser.add_argument('--plotOnly', dest='plotOnly', action='store_true',
                        default=False, help="Reload the metric values from disk and re-plot them.")
    args = parser.parse_args()

    if args.runName is None:
        args.runName = os.path.basename(args.dbfile).replace('_sqlite.db', '')
        args.runName = args.runName.replace('.db', '')
    if args.outDir is None:
        args.outDir = os.path.join(args.runName, subdir)
    return args


if __name__ == '__main__':
    args = parseArgs()
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()