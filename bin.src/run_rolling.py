#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    bdict = {}
    plotbundles = []

    # Run the timeHealpix batch - creates "uniformity over time as function of healpix" plots
    uniformity, pb = batches.pixelTime(colmap=colmap, runName=args.runName,
                                       extraSql=None, extraMetadata=None)
    bdict.update(uniformity)
    plotbundles.append(pb)

    return bdict


def run(bdict, opsdb, colmap, args):
    resultsDb = db.ResultsDb(outDir=args.outDir)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb,
                                 saveEarly=False)
    group.runAll()
    group.plotAll()
    resultsDb.close()
    mafUtils.writeConfigs(opsdb, args.outDir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run metrics for time distribution.")
    args = parseArgs(subdir='rolling_check', parser=parser)
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        print('Cannot replot these metrics.')
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
