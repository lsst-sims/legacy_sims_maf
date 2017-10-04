#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *

def setBatches(opsdb, colmap, args):
    bdict = {}
    # number of observations per propsosal and per night.
    bdict.update(batches.nvisitsPerProp(opsdb, colmap, args.runName))
    # add nvisits / teff maps.
    propids, proptags, sqltags = setSQL(opsdb)
    # All props
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName))
    # WFD only
    bdict.update(batches.nvisitsM5Maps(colmap, args.runName,
                                       extraSql=sqltags['WFD'], extraMetadata='WFD'))
    # All props.
    bdict.update(batches.tEffMetrics(colmap, args.runName))
    # WFD only.
    bdict.update(batches.tEffMetrics(colmap, args.runName,
                                     extraSql=sqltags['WFD'], extraMetadata='WFD'))
    return bdict
    return bdict


if __name__ == '__main__':
    args = parseArgs('nvisits')
    opsdb, colmap = connectDb(args.dbfile)
    bdict = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()
