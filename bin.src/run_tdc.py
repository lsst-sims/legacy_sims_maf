#!/usr/bin/env python

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import numpy as np
import healpy as hp

import lsst.sims.maf.db as db
import lsst.sims.maf.batches as batches
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.utils as mafUtils

if __name__ == '__main__':
    subdir='science'
    parser = argparse.ArgumentParser()
    parser.add_argument("dbfile", type=str, help="Sqlite file of observations (full path).")
    parser.add_argument("--runName", type=str, default=None, help="Run name."
                                                                  "Default is based on dbfile name.")
    parser.add_argument("--outDir", type=str, default=None, help="Output directory."
                                                                 "Default is runName/%s." % (subdir))
    args = parser.parse_args()

    if args.runName is None:
        args.runName = os.path.basename(args.dbfile).replace('_sqlite.db', '')
        args.runName = args.runName.replace('.db', '')
    if args.outDir is None:
        args.outDir = os.path.join(args.runName, subdir)

    opsdb = db.OpsimDatabase(args.dbfile)
    resultsDb = db.ResultsDb(outDir=args.outDir)
    
    colmap = batches.ColMapDict()
    
    
    # Run science radar, plot immediately and do not write to disk
    #bdict = batches.scienceRadarBatch(colmap=colmap, runName=args.runName,
    #                                  extraSql=None, extraMetadata=None,
    #                                  nside=64, benchmarkArea=18000, benchmarkNvisits=825, DDF=True)
    
    #group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, saveEarly=False)
    #group.runAll(clearMemory=True, plotNow=True)
    

    # Run but wait to plot the TDC batch (and save so we can pull out summary vals later)
    bdict = batches.tdcBatch(runName=args.runName)
    group = mb.MetricBundleGroup(bdict, opsdb, outDir=args.outDir, resultsDb=resultsDb, saveEarly=True)
    group.runAll(plotNow=False)

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

    group.plotAll()

    # calculate TDC summary values
    accuracy_threshold = 1.0
    precision_threshold = 5.0
    results = {}
    keys = ['high_accuracy_fraction', 'high_accuracy_area', 'total_area',
            'hA_cadence', 'hA_season', 'hA_campaign', 'hA_rate',
            'precision_per_lens', 'N_lenses', 'distance_precision']
    # reference the (full) TDC metric bundle
    bundle = bdict[f'{args.runName.replace(".", "_")}_TDC_HEAL']
    
    # Sigh .. a complex metric means each value is a dictionary ..
    x = bundle.metricValues.compressed()
    f = np.array([each['rate'] for each in x])
    A = np.array([each['accuracy'] for each in x])
    P = np.array([each['precision'] for each in x])
    c = np.array([each['cadence'] for each in x])
    s = np.array([each['season'] for each in x])
    y = np.array([each['campaign'] for each in x])
    
    high_accuracy = np.where((A < accuracy_threshold) & (P < precision_threshold))
    
    nside = bundle.slicer.nside
    pix_area = hp.nside2pixarea(nside, degrees=True)
    
    frac = 100*(1.0*len(A[high_accuracy]))/(1.0*len(A))
    results['high_accuracy_fraction'] = frac
    
    hAarea = len(high_accuracy[0]) * pix_area
    results['high_accuracy_area'] = hAarea
    area = len(A) * pix_area
    results['total_area'] = area
    
    hAcadence = np.median(c[high_accuracy])
    results['hA_cadence'] = hAcadence
    hAseason = np.median(s[high_accuracy])
    results['hA_season'] = hAseason
    hAcampaign = np.median(y[high_accuracy])
    results['hA_campaign'] = hAcampaign
    
    precision_per_lens = np.array([np.mean(P[high_accuracy]), 4.0])
    precision_per_lens = np.sqrt(np.sum(precision_per_lens * precision_per_lens))
    results['precision_per_lens'] = precision_per_lens
    
    fraction_found = np.mean(f[high_accuracy])
    results['hA_rate'] = fraction_found
    n_lenses = int((hAarea / 18000.0) * (fraction_found / 30.0) * 400)
    results['N_lenses'] = n_lenses
    
    distance_precision = precision_per_lens / np.sqrt(n_lenses)
    results['distance_precision'] = distance_precision
    # Write these summary values to the resultsDb
    metricId = resultsDb.updateMetric(bundle.metric.name, bundle.slicer.slicerName,
                                      args.runName, bundle.constraint, bundle.metadata, None)
    for k in results:
        resultsDb.updateSummaryStat(metricId, summaryName=k, summaryValue=results[k])
        
    # Write config to disk.
    mafUtils.writeConfigs(opsdb, args.outDir)

    resultsDb.close()
    opsdb.close()
