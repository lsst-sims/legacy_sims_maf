import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.utils as utils
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binners as binners
import lsst.sims.maf.binMetrics as binMetrics

rng = np.random.RandomState(800233)

oo = db.OpsimDatabase('sqlite:///opsimblitz1_1131_sqlite.db')

cols = ['expMJD', 'fieldRA', 'fieldDec', 'hexdithra', 'hexdithdec']
simdata = oo.fetchMetricData(cols, '')

nside = 128
binner = binners.HealpixBinner(nside=nside)
binner.setupBinner(simdata)
hexbinner = binners.HealpixBinner(nside=nside, spatialkey1='hexdithra', spatialkey2='hexdithdec')
hexbinner.setupBinner(simdata)

# Generate random values over the entire sky
randomallskymetricval = ma.MaskedArray(data = rng.rand(len(binner)),
                                       mask = np.zeros(len(binner), bool), 
                                       fill_value= binner.badval)

onesallskymetricval = ma.MaskedArray(data = np.ones(len(binner), float),
                                     mask = np.zeros(len(binner), bool), 
                                     fill_value= binner.badval)

# Generate random values over area with LSST observations
gm = binMetrics.BaseBinMetric()
gm.setBinner(binner)
metric = metrics.BinaryMetric('expMJD')
gm.setMetrics([metric])
gm.runBins(simdata, 'test')

hexgm = binMetrics.BaseBinMetric()
hexgm.setBinner(hexbinner)
hexgm.setMetrics([metric])
hexgm.runBins(simdata, 'test')

randomnodithermetricval = ma.MaskedArray(data = rng.rand(len(binner)),
                                         mask = gm.metricValues[gm.metricNames[0]].mask,
                                         fill_value = binner.badval)
randomhexdithermetricval = ma.MaskedArray(data = rng.rand(len(binner)),
                                          mask = hexgm.metricValues[hexgm.metricNames[0]].mask,
                                          fill_value = hexbinner.badval)


onesnodithermetricval = ma.MaskedArray(data = np.ones(len(binner), float),
                                       mask = gm.metricValues[gm.metricNames[0]].mask,
                                       fill_value = binner.badval)
oneshexdithermetricval = ma.MaskedArray(data = np.ones(len(binner), float),
                                        mask = hexgm.metricValues[hexgm.metricNames[0]].mask,
                                        fill_value = hexbinner.badval)

binner.plotSkyMap(randomallskymetricval, title='All Sky')
binner.plotSkyMap(randomnodithermetricval, title='No dither')
binner.plotSkyMap(randomnodithermetricval, title='Hex dither')
binner.plotSkyMap(onesallskymetricval, title='All Sky')
binner.plotSkyMap(onesnodithermetricval, title='No dither')
binner.plotSkyMap(onesnodithermetricval, title='Hex dither')


fignum = binner.plotPowerSpectrum(randomallskymetricval, label='All Sky')
fignum = binner.plotPowerSpectrum(randomnodithermetricval, label='No Dither', fignum=fignum)
fignum = binner.plotPowerSpectrum(randomhexdithermetricval, label='Hex Dither', fignum=fignum)
fignum = binner.plotPowerSpectrum(randomnodithermetricval, label='No Dither, w/Dipole', fignum=fignum, removeDipole=False)
fignum = binner.plotPowerSpectrum(randomhexdithermetricval, label='Hex Dither, w/Dipole', fignum=fignum, removeDipole=False, 
                                  addLegend=True, legendloc = 'lower right',
                                  title='Random Metric, Healpix grid nside=%d' %(nside))

fignum = binner.plotPowerSpectrum(onesallskymetricval, label='All Sky')
fignum = binner.plotPowerSpectrum(onesnodithermetricval, label='No Dither', fignum=fignum)
fignum = binner.plotPowerSpectrum(oneshexdithermetricval, label='Hex Dither', fignum=fignum)
fignum = binner.plotPowerSpectrum(onesnodithermetricval, label='No Dither, w/Dipole', fignum=fignum, removeDipole=False)
fignum = binner.plotPowerSpectrum(oneshexdithermetricval, label='Hex Dither, w/Dipole', fignum=fignum, removeDipole=False, 
                                  addLegend=True, legendloc='lower right',
                                  title='Ones (constant) Metric, Healpix grid nside=%d' %(nside))

plt.show()
