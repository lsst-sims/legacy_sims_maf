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
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.batches as batches
import lsst.sims.maf.utils as mafUtils




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
    elif version == "FBS":
        opsdb = db.OpsimDatabaseFBS(dbfile)
        colmap = batches.ColMapDict('FBS')
    return opsdb, colmap


def setSQL(opsdb, sqlConstraint=None, extraMeta=None):
    # Fetch the proposal ID values from the database
    # If there is no proposal database, propids and proptags will be empty dictionaries.
    propids, proptags = opsdb.fetchPropInfo()
    sqltags = {'All': sqlConstraint}
    metadata = {'All': ''}
    if 'WFD' in proptags:
        # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
        wfdWhere = opsdb.createSQLWhere('WFD', proptags)
        sqltags['WFD'] = wfdWhere
        metadata['WFD'] = 'WFD'
    if 'DD' in proptags:
        ddWhere = opsdb.createSQLWhere('DD', proptags)
        sqltags['DD'] = ddWhere
        metadata['DD'] = 'DD'
    if 'NES' in proptags:
        nesWhere = opsdb.createSQLWhere('NES', proptags)
        sqltags['NES'] = nesWhere
        metadata['NES'] = 'NES'
    if sqlConstraint is not None:
        wfdWhere = '(%s) and (%s)' % (sqlConstraint, wfdWhere)
        ddWhere = '(%s) and (%s)' % (sqlConstraint, ddWhere)
        nesWhere = '(%s) and (%s)' % (sqlConstraint, nesWhere)
    if sqlConstraint is not None and len(sqlConstraint) > 0:
        md = sqlConstraint.replace('=', '').replace('filter', '').replace("'", '')
        md = md.replace('"','').replace('  ', ' ')
        for t in metadata:
            metadata[t] += ' %s' % md
    if extraMeta is not None and len(extraMeta) > 0:
        for t in metadata:
            metadata[t] += ' %s' % extraMeta
    # Reset metadata to None if there was nothing there. (helpful for batches).
    for t in metadata:
        if len(metadata[t]) == 0:
            metadata[t] = None
    return (propids, proptags, sqltags, metadata)


def run(bdict, opsdb, colmap, args, save=True):
    resultsDb = db.ResultsDb(outDir=args.outDir)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, saveEarly=False)
    if save:
        group.runAll()
        group.plotAll()
    else:
        group.runAll(clearMemory=True, plotNow=True)
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
    parser.add_argument('--sqlConstraint', type=str, default=None,
                        help="SQL constraint to apply to all metrics. "
                             " e.g.: 'night <= 365' or 'propId = 5' "
                             " (**may not work with slew batches)")
    parser.add_argument("--nyears", type=int, default=10, help="Number of years in the run (default 10).")
    parser.add_argument("--ditherStacker", type=str, default=None,
                        help="Name of dither stacker to use for RA/Dec, e.g. 'RandomDitherPerNightStacker'."
                        " Note that this currently is only applied to the run_srd script. ")
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
