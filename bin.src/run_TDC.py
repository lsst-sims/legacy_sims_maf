import os
import glob
import argparse
import matplotlib
matplotlib.use("TkAgg")

import lsst.sims.maf.db as db
import lsst.sims.maf.batches as batches
import lsst.sims.maf.metricBundles as mb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    if args.db is None:
        if os.path.isfile('trackingDb_sqlite.db'):
            os.remove('trackingDb_sqlite.db')
        db_files = glob.glob('*.db')
    else:
        db_files = [args.db]

    run_names = [os.path.basename(name).replace('.db', '') for name in db_files]

    for filename, name in zip(db_files, run_names):
        opsdb = db.OpsimDatabase(filename)
        colmap = batches.ColMapDict()
        bdict = batches.tdcBatch(runName=name)
        group = mb.MetricBundleGroup(bdict, opsdb, outDir=name + '_tdc')
        group.runAll()

        minVal = 0.01
        maxVal = {'Accuracy': 0.2, 'Precision': 20.0, 'Rate': 40, 'Cadence': 14, 'Season': 8.0,
                  'Campaign': 11.0}
        units = {'Accuracy': '%', 'Precision': '%', 'Rate': '%', 'Cadence': 'days', 'Season': 'months',
                 'Campaign': 'years'}
        for key in maxVal:
            plotDict = {'xMin': minVal, 'xMax': maxVal[key], 'colorMin': minVal, 'colorMax': maxVal[key]}
            plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
            bdict['TDC_%s' % (key)].setPlotDict(plotDict)
        for key in maxVal:
            plotDict = {'xMin': minVal, 'xMax': maxVal[key], 'colorMin': minVal, 'colorMax': maxVal[key]}
            plotDict['xlabel'] = 'TDC %s (%s)' % (key, units[key])
            bdict['TDC_%s' % (key)].setPlotDict(plotDict)

        group.plotAll(closefigs=True)

        opsdb.close()
