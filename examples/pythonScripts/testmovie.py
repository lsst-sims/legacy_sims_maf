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
night = 3610
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
i = 0

# Recreate the work done in opsimMovie.py runSlices method (but for a single slice/frame).
ms = movieslicer[i]
slicenumber = sliceformat %(i)

simdatasubset = simdata[ms['idxs']]
ops = mm.setupOpsimFieldSlicer(simdatasubset, fields)
tstep = ms['slicePoint']['binRight'] - bins[i]
if tstep > 1:
    tstep = 40./24./60./60.
# Convert expMJD days to time from noon on first day. (local midnight is at 0.16)
times_from_start = ms['slicePoint']['binRight'] - (int(bins[0]) + 0.16 - 0.5)
years = int(times_from_start/365)
days = times_from_start - years*365
plotlabel = 'Year %d Day %.4f' %(years, days)

metricList = mm.setupMetrics(opsimName, metadata, plotlabel,
                             t0=ms['slicePoint']['binRight'], tStep=tstep, years=years, onlyVisitFilters=True)
sm = sliceMetrics.RunSliceMetric(outDir = '.', useResultsDb=False, figformat='png', dpi=72, thumbnail=False)
sm.setSlicer(ops)
sm.setMetrics(metricList)
sm.runSlices(simdatasubset, simDataName=opsimName)

visitNow = np.where(simdatasubset['expMJD'] == simdatasubset['expMJD'].max())[0]
raCen = simdatasubset['lst'][visitNow][0]
lat_tele = np.radians(-29.666667)
horizonlon, horizonlat = mm.addHorizon(lat_telescope=lat_tele, raCen=raCen)
for mId in sm.metricValues:
    fignum = None
    fignum = ops.plotSkyMap(sm.metricValues[mId], fignum=fignum, raCen=raCen, **sm.plotDicts[mId])
    ax = plt.gca()
    # Add horizon and zenith.
    plt.plot(horizonlon, horizonlat, 'k.', alpha=0.3, markersize=1.8)
    plt.plot(0, lat_tele, 'k+')
    if sm.metricNames[mId] == 'VisitFilters':
        # Add the time stamp info (plotlabel) with a fancybox.
        plt.figtext(0.75, 0.9, '%s' %(plotlabel), bbox=dict(boxstyle='Round, pad=0.7', fc='w', ec='k', alpha=0.5))
        # Add a legend.
        filterstacker = FilterColorStacker()
        for j, f in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            plt.figtext(0.92, 0.55 - j*0.035, f, color=filterstacker.filter_rgb_map[f])
        moonRA = np.mean(simdatasubset['moonRA'][visitNow])
        lon = -(moonRA - raCen - np.pi) % (np.pi*2) - np.pi
        moonDec = np.mean(simdatasubset['moonDec'][visitNow])
        # Note that moonphase is 0-100
        moonPhase = np.mean(simdatasubset['moonPhase'][visitNow])/100.
        alpha = np.max([moonPhase, 0.15])
        circle = Circle((lon, moonDec), radius=0.05, color='k', alpha=alpha)
        ax.add_patch(circle)
        # Add some explanatory text.
        ecliptic = Line2D([], [], color='r', label="Ecliptic plane")
        galaxy = Line2D([], [], color='b', label="Galactic plane")
        horizon = Line2D([], [], color='k', alpha=0.3, label="20 deg elevation limit")
        moon = Line2D([], [], color='k', linestyle='', marker='o', markersize=8, alpha=alpha,
                              label="\nMoon (Dark=Full)\n         (Light=New)")
        zenith = Line2D([], [], color='k', linestyle='', marker='+', markersize=5, label="Zenith")
        plt.legend(handles=[horizon, zenith, galaxy, ecliptic, moon], loc=[0.1, -0.35], ncol=3, frameon=False,
                title = 'Aitoff plot showing HA/Dec of simulated survey pointings', numpoints=1, fontsize='small')
    plt.savefig(os.path.join('.', sm.metricNames[mId]  + '_'  + slicenumber + '_SkyMap.png'), format='png')



plt.show()
