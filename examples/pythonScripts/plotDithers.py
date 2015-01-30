import numpy as np
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.stackers as stackers

# Database and sqlconstraint info.
dbAddress = 'sqlite:////Users/lynnej/workshop/ops2_1075_sqlite.db'
# Choose a field .. 2634 is at 0.545/-0.004
sqlconstraint = 'fieldID = 2634'

# Get a field, along with its series of observations, from opsim.
oo = db.OpsimDatabase(dbAddress)
simdata = oo.fetchMetricData(colnames=['fieldRA', 'fieldDec', 'expMJD', 'night'], sqlconstraint=sqlconstraint)

stackerDict = {}
stackerDict['RandomDither'] = stackers.RandomDitherStacker()
stackerDict['NightlyRandomDither'] = stackers.NightlyRandomDitherStacker()
stackerDict['SpiralDither'] = stackers.SpiralDitherStacker()
stackerDict['NightlySpiralDither'] = stackers.NightlySpiralDitherStacker()
stackerDict['SequentialHexDither'] = stackers.SequentialHexDitherStacker()
stackerDict['NightlySequentialHexDither'] = stackers.NightlySequentialHexDitherStacker()

stepsize= np.pi/50.
theta = np.arange(0, np.pi*2.+stepsize, stepsize)
radius = np.radians(1.75)
for s in stackerDict:
    print s
    simdata = stackerDict[s].run(simdata)
    plt.figure()
    plt.axis('equal')
    plt.plot(simdata['fieldRA'][0], simdata['fieldDec'][0], 'g+')
    plt.plot(radius*np.cos(theta)+simdata['fieldRA'][0], radius*np.sin(theta)+simdata['fieldDec'][0], 'g-')
    print stackerDict[s].colsAdded[0], stackerDict[s].colsAdded[1]
    plt.plot(simdata[stackerDict[s].colsAdded[0]], simdata[stackerDict[s].colsAdded[1]], 'k-', alpha=0.2)
    plt.plot(simdata[stackerDict[s].colsAdded[0]], simdata[stackerDict[s].colsAdded[1]], 'r.')
    plt.title(s)
    plt.savefig('%s.png' %(s))

