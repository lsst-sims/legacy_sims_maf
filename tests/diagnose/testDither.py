import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.sliceMetrics as sliceMetrics


oo = db.OpsimDatabase('sqlite:///../opsimblitz1_1131_sqlite.db')

cols = ['fieldID', 'fieldRA', 'fieldDec']
simdata = oo.fetchMetricData(cols, '')
fielddata = oo.fetchFieldsFromFieldTable()

# Add dither column
randomdither = utils.RandomDither(maxDither=1.8, randomSeed=42)
simdata = randomdither.run(simdata)

# Add columns showing the actual dither values
# Note that because RA is wrapped around 360, there will be large values of 'radith' near this point
basestacker = utils.BaseStacker()
basestacker.colsAdded = ['radith', 'decdith']
simdata = basestacker._addStackers(simdata)
simdata['radith'] = simdata['randomRADither'] - simdata['fieldRA']
simdata['decdith'] = simdata['randomDecDither'] - simdata['fieldDec']


metriclist = []
metriclist.append(metrics.MeanMetric('radith'))
metriclist.append(metrics.MeanMetric('decdith'))
metriclist.append(metrics.RmsMetric('radith'))
metriclist.append(metrics.RmsMetric('decdith'))
metriclist.append(metrics.FullRangeMetric('radith'))
metriclist.append(metrics.FullRangeMetric('decdith'))
metriclist.append(metrics.MaxMetric('decdith'))
metriclist.append(metrics.MinMetric('decdith'))

slicer = slicers.OpsimFieldSlicer()
slicer.setupSlicer(simdata, fielddata)

gm = sliceMetrics.BaseSliceMetric()
gm.setSlicer(slicer)
gm.setMetrics(metriclist)
gm.runSlices(simdata, 'Dither Test')
gm.plotAll(savefig=False)

plt.show()
