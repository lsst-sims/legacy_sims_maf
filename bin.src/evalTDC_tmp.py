import os
import glob
import numpy as np
import healpy as hp
import pandas as pd

import lsst.sims.maf.metricBundles as mb


if __name__ == '__main__':

    # Read all TDC npz outputs
    files = glob.glob('*/*TDC_HEAL.npz')
    runs = [os.path.basename(i).replace('_TDC_HEAL.npz', '') for i in files]

    accuracy_threshold = 1.0
    precision_threshold = 5.0
    results = {}
    keys = ['run', 'high_accuracy_fraction', 'high_accuracy_area', 'total_area',
            'hA_cadence', 'hA_season', 'hA_campaign', 'hA_rate',
            'precision_per_lens', 'N_lenses', 'distance_precision']
    for k in keys:
        results[k] = []
    for f, r in zip(files, runs):
        print(f'working on {f}')
        # Read metric results from disk
        bundle = mb.createEmptyMetricBundle()
        bundle.read(f)
        results['run'].append(bundle.runName)

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
        results['high_accuracy_fraction'].append(frac)

        hAarea = len(high_accuracy[0]) * pix_area
        results['high_accuracy_area'].append(hAarea)
        area = len(A) * pix_area
        results['total_area'].append(area)

        hAcadence = np.median(c[high_accuracy])
        results['hA_cadence'].append(hAcadence)
        hAseason = np.median(s[high_accuracy])
        results['hA_season'].append(hAseason)
        hAcampaign = np.median(y[high_accuracy])
        results['hA_campaign'].append(hAcampaign)

        precision_per_lens = np.array([np.mean(P[high_accuracy]), 4.0])
        precision_per_lens = np.sqrt(np.sum(precision_per_lens * precision_per_lens))
        results['precision_per_lens'].append(precision_per_lens)

        fraction_found = np.mean(f[high_accuracy])
        results['hA_rate'] = fraction_found
        n_lenses = int((hAarea / 18000.0) * (fraction_found / 30.0) * 400)
        results['N_lenses'].append(n_lenses)

        distance_precision = precision_per_lens / np.sqrt(n_lenses)
        results['distance_precision'].append(distance_precision)

    df = pd.DataFrame(results)
    df.to_csv('tdc_summary.csv', index=False)