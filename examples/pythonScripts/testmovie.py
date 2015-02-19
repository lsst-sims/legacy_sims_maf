# Useful test script to generate a single frame of an opsimMovie.py movie.
# For diagnosing problems/changes. Replicates what happens in opsimMovie.py, but has to do some things explicitly here.

import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import lsst.sims.maf.db as db
from lsst.sims.maf.stackers import FilterColorStacker
import lsst.sims.maf.sliceMetrics as sliceMetrics
import opsimMovie as mm

# Set database information.
opsimName = 'lucy_1002'
dbAddress = 'sqlite:///' + opsimName + '_sqlite.db'

# Choose night to look at.
night = 0
sqlconstraint = 'night<=%d' %(night)
metadata = 'night %d' %(night)

# Get observations from the particular night, plus previous.
oo = db.OpsimDatabase(dbAddress)
simdata, fields = mm.getData(oo, sqlconstraint)
condition = np.where(simdata['night'] == night)[0]

# Set bins for movieslicer.
bins = simdata['expMJD'][condition]
bins[0] = simdata['expMJD'].min()
movieslicer = mm.setupMovieSlicer(simdata, bins)
sliceformat = '%s0%dd' %('%', int(np.log10(len(movieslicer)))+1)

# Choose frame to plot. (len(movieslicer)-1 will be the last frame).
i = len(movieslicer)-1

# Recreate the work done in opsimMovie.py runSlices method (but for a single slice/frame).
ms = movieslicer[i]
slicenumber = sliceformat %(i)
time = ms['slicePoint']['binRight']

simdatasubset = simdata[ms['idxs']]
ops = mm.setupOpsimFieldSlicer(simdatasubset, fields)
tstep = ms['slicePoint']['binRight'] - bins[i]
if tstep > 1:
    tstep = 40./24./60./60.
metric = mm.setupMetrics(opsimName, metadata, t0=time, tStep=tstep)
# Convert expMJD days to time from noon on first day. (local midnight is at 0.16)
times_from_start = ms['slicePoint']['binRight'] - (int(bins[0]) + 0.16 - 0.5)
years = int(times_from_start % 365)
days = times_from_start - years*365 - 0.5 + 0.16
metric.plotDict['label'] = 'Year %d Day %.4f' %(years, days)
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
# Add some explanatory text.
ecliptic = Line2D([], [], color='r', label="Ecliptic Plane")
galaxy = Line2D([], [], color='b', label="Galactic Plane")
horizon = Line2D([], [], color='k', alpha=0.3, label="Elevation limit")
moon = Line2D([], [], color='k', linestyle='', marker='o', markersize=8, alpha=alpha, label="Moon")
plt.legend(handles=[horizon, galaxy, ecliptic, moon], loc=[0.05, -0.3], ncol=4, frameon=False,
           title = 'Aitoff plot showing HA/Dec of simulated survey pointings', numpoints=1, fontsize='small')
plt.savefig(os.path.join('.', 'movieFrame_' + slicenumber + '_SkyMap.png'), format='png')

fig = plt.figure(fignum)

plt.show()
