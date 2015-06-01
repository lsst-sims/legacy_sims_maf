#! /usr/bin/env python
# A little script to time out the penalty for running chip gaps

import timeit


def runChips(useCamera=False):
    import numpy as np
    import lsst.sims.maf.slicers as slicers
    import lsst.sims.maf.metrics as metrics
    import lsst.sims.maf.metricBundles as metricBundles
    import lsst.sims.maf.db as db
    from lsst.sims.maf.plots import PlotHandler
    import matplotlib.pylab as plt
    import healpy as hp


    print 'Camera setting = ', useCamera

    dbAddress = 'sqlite:///enigma_1189_sqlite.db'
    sqlWhere = 'filter = "r" and night < 800 and fieldRA < %f and fieldDec > %f and fieldDec < 0' % (np.radians(15), np.radians(-15))
    opsdb = db.OpsimDatabase(dbAddress)
    outDir = 'Camera'
    resultsDb = db.ResultsDb(outDir=outDir)

    nside=512
    tag = 'F'
    if useCamera:
        tag='T'
    metric = metrics.CountMetric('expMJD', metricName='chipgap_%s'%tag)

    slicer = slicers.HealpixSlicer(nside=nside, useCamera=useCamera)
    bundle1 = metricBundles.MetricBundle(metric,slicer,sqlWhere)

    bg = metricBundles.MetricBundleGroup({0:bundle1},opsdb, outDir=outDir, resultsDb=resultsDb)
    bg.runAll()
    hp.gnomview(bundle1.metricValues, xsize=800,ysize=800, rot=(7,-7,0), unit='Count', min=1,max=21)
    plt.savefig(outDir+'/fig'+tag+'.png')


if __name__ == "__main__":


    t1 = timeit.timeit("runChips()", setup="from __main__ import runChips", number=1)
    t2 = timeit.timeit("runChips(useCamera=True)", setup="from __main__ import runChips", number=1)

    print '--------'
    print 'time without chips = %f'%t1
    print 'time with chips = %f'%t2

# Results:
#--------
#time without chips = 80.819745
#time with chips = 92.150030
