#!/usr/bin/env python

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import lsst.sims.maf.batches as batches
from run_generic import *


def setBatches(opsdb, colmap, args):
    # Identify proposal information, etc.
    propids, proptags, sqls, metadata = setSQL(opsdb, args.sqlConstraint)

    bdict = {}
    plotbundles = []
    # Intranight (pairs/time)
    intranight_all, plots = batches.intraNight(colmap, args.runName, extraSql=args.sqlConstraint)
    bdict.update(intranight_all)
    plotbundles.append(plots)
    sql = '(%s) or (%s)' % (sqls['WFD'], sqls['NES'])
    md = 'WFD+' + metadata['NES']
    intranight_wfdnes, plots = batches.intraNight(colmap, args.runName, extraSql=sql,
                                                  extraMetadata=md)
    bdict.update(intranight_wfdnes)
    plotbundles.append(plots)
    internight_all, plots = batches.interNight(colmap, args.runName, extraSql=args.sqlConstraint)
    bdict.update(internight_all)
    plotbundles.append(plots)
    internight_wfd, plots = batches.interNight(colmap, args.runName, extraSql=sqls['WFD'],
                                               extraMetadata=metadata['WFD'])
    bdict.update(internight_wfd)
    plotbundles.append(plots)
    return bdict, plots


if __name__ == '__main__':
    args = parseArgs('cadence')
    opsdb, colmap = connectDb(args.dbfile)
    bdict, plotbundles = setBatches(opsdb, colmap, args)
    if args.plotOnly:
        replot(bdict, opsdb, colmap, args)
    else:
        run(bdict, opsdb, colmap, args)
    opsdb.close()

