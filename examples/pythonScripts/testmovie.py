# Useful test script to generate a single frame of an opsimMovie.py movie.
# For diagnosing problems/changes. Replicates what happens in opsimMovie.py, but has to do some things explicitly here.

import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import lsst.sims.maf.db as db
from lsst.sims.maf.stackers import FilterColorStacker
import lsst.sims.maf.sliceMetrics as sliceMetrics
import opsimMovie as mm

dbAddress = 'sqlite:///lucy_1002_sqlite.db'
night = 0
sqlconstraint = 'night<=%d' %(night)
opsimName = 'lucy_1002'
metadata = 'night %d' %(night)

oo = db.OpsimDatabase(dbAddress)
# Get observations from the particular night, plus previous.
simdata, fields = mm.getData(oo, sqlconstraint)
condition = np.where(simdata['night'] == night)[0]
bins = simdata['expMJD'][condition]
bins[0] = simdata['expMJD'].min()

movieslicer = mm.setupMovieSlicer(simdata, bins)
sliceformat = '%04d'


i = len(movieslicer)-1
i = 2

ms = movieslicer[i]
slicenumber = sliceformat %(i)
time = ms['slicePoint']['binRight']

simdatasubset = simdata[ms['idxs']]
ops = mm.setupOpsimFieldSlicer(simdatasubset, fields)
tstep = ms['slicePoint']['binRight'] - bins[i]
if tstep > 1:
    tstep = 40./24./60./60.
metric = mm.setupMetrics(opsimName, metadata, t0=time, tStep=tstep)
sm = sliceMetrics.RunSliceMetric(outDir = '.', useResultsDb=False, figformat='png', dpi=72, thumbnail=False)
sm.setSlicer(ops)
sm.setMetrics([metric])
sm.runSlices(simdatasubset, simDataName=opsimName)

visitNow = np.where(simdatasubset['expMJD'] == simdatasubset['expMJD'].max())[0]
raCen = simdatasubset['lst'][visitNow][0]
fignum = None
fignum = ops.plotSkyMap(sm.metricValues[0], fignum=fignum, raCen=raCen, **sm.plotDicts[0])
ax = plt.gca()
# Add a legend.
filterstacker = FilterColorStacker()
for j, f in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
    plt.figtext(0.92, 0.55 - j*0.035, f, color=filterstacker.filter_rgb_map[f])
moonRA = np.mean(simdatasubset['moonRA'][visitNow])
lon = -(moonRA - raCen - np.pi) % (np.pi*2) - np.pi
moonDec = np.mean(simdatasubset['moonDec'][visitNow])
# Note that moonphase is 0-100
moonPhase = np.mean(simdatasubset[visitNow]['moonPhase'])/100.
alpha = np.max([moonPhase, 0.15])
circle = Circle((lon, moonDec), radius=0.05, color='k', alpha=alpha)
ax.add_patch(circle)
# Add horizon and zenith.
lat_tele = np.radians(-29.666667)
lon, lat = mm.addHorizon(lat_telescope=lat_tele, raCen=raCen)
plt.plot(lon, lat, 'k.', alpha=0.3, markersize=1.8)
plt.plot(0, lat_tele, 'k+')

plt.savefig(os.path.join('.', 'movieFrame_' + slicenumber + '_SkyMap.png'), format='png')

fig = plt.figure(fignum)

plt.show()
