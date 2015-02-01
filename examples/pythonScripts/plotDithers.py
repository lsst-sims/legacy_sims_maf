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
#radius = np.radians(1.75)
radius = 1.75
for s in stackerDict:
    print s
    simdata = stackerDict[s].run(simdata)
    plt.figure()
    plt.axis('equal')
    x = np.degrees(simdata['fieldRA'][0])
    y = np.degrees(simdata['fieldDec'][0])
    plt.plot(x, y, 'g+')
    plt.plot(radius*np.cos(theta)+x, radius*np.sin(theta)+y, 'g-')
    print stackerDict[s].colsAdded[0], stackerDict[s].colsAdded[1]
    x = np.degrees(simdata[stackerDict[s].colsAdded[0]])
    y = np.degrees(simdata[stackerDict[s].colsAdded[1]])
    plt.plot(x, y, 'k-', alpha=0.2)
    plt.plot(x, y, 'r.')
    plt.title(s)
    plt.savefig('%s.png' %(s))

