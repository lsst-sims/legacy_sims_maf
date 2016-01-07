import os, argparse
import numpy as np
from lsst.sims.maf.viz import MafRunComparison

def pandaprint(stats):
    for i in range(len(stats)):
        writestring = ''
        for j in range(len(stats[i])):
            writestring += ' %s;' %stats[i][j]
        print writestring.lstrip(' ')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Python script to run simple comparison between multiple opsim runs.')
    parser.add_argument('--baseDir', type=str, default=None, help='Root directory containing file with list of runs and top directory containing MAF results.')
    parser.add_argument('--runlist', type=str, default=None, help='File containing the names of the runs to compare '
                        '(and optionally the directories, relative to baseDir, where they reside).')
    parser.set_defaults()
    args = parser.parse_args()

    baseDir = args.baseDir

    # Read the runs to be compared from a file called 'tier1.txt'
    f = open(os.path.join(baseDir, args.runlist), 'r')
    runlist = []
    rundirs = []
    for line in f:
        runlist.append(line.split()[0])
        if len(line.split()) > 1:
            rundirs.append(line.split()[1])
        else:
            rundirs.append(line.split()[0])

    runCompare = MafRunComparison(baseDir = baseDir, runlist = runlist, rundirs = rundirs)


    writestring = 'Summary Name; '
    for r in runlist:
        writestring += '%s; ' %r
    print writestring

    # Get 'overview' statistics.

    # Total number of visits
    metricName = 'NVisits'
    metricMetadata = 'All Visits'
    slicerName = 'UniSlicer'
    summaryName = 'Count'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Total open shutter time (in megasec)
    # Need to add this to MAF

    # Percentage of visits for each proposal
    metricName = 'NVisits'
    metricMetadata = None
    slicerName = 'UniSlicer'
    summaryName = 'Fraction of total'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Mean Surveying efficiency (??)
    metricName = 'Total effective time of survey'
    metricMetadata = 'All Visits'
    slicerName = None
    summaryName = '(days)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median open shutter fraction
    metricName = 'OpenShutterFraction'
    metricMetadata = 'Per night'
    slicerName = None
    summaryName = 'Median'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Mean and Median number of visits per field
    metricName = 'NVisits'
    slicerName = 'OpsimFieldSlicer'
    for summaryName in ('Mean', 'Median'):
        for f in ('u', 'g', 'r', 'i', 'z', 'y'):
            metricMetadata = '%s band, all props' %f
            summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
            pandaprint(summary)

    # Median number of visits per night
    metricName = 'NVisits'
    metricMetadata = 'Per night'
    slicerName = 'OneDSlicer'
    summaryName = 'Median'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Number of nights with observations
    metricName = 'Nights with observations'
    metricMetadata = 'All Visits'
    slicerName = 'UniSlicer'
    summaryName = '(days)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Mean slew time
    metricName = 'Mean slewTime'
    slicerName = None
    metricMetadata = None
    summaryName = None
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # fO NV and Area
    metricName = 'fO'
    metricMetadata = 'All Visits (non-dithered)'
    slicerName = None
    summaryName = 'fONv: Area (sqdeg)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)
    summaryName = 'fOArea: Nvisits (#)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median r band seeing
    metricName = 'Median finSeeing'
    metricMetadata = 'r band, all props'
    slicerName = None
    summaryName = 'Identity'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median r band airmass
    metricName = 'Median airmass'
    metricMetadata = 'r band, all props'
    slicerName = 'UniSlicer'
    summaryName = 'Identity'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median proper motion accuracy @20
    metricName = 'Proper Motion 20'
    metricMetadata = None
    slicerName = None
    summaryName = 'Median'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median proper motion accuracy @24
    metricName = 'Proper Motion 24'
    metricMetadata = None
    slicerName = None
    summaryName = 'Median'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # WFD performance metrics

    # Median single visit depth in ugrizy (all visits)
    metricName = 'Median fiveSigmaDepth'
    slicerName = 'UniSlicer'
    metricMetadata = []
    summaryName = 'Identity'
    for f in ('u', 'g', 'r', 'i', 'z', 'y'):
        metricMetadata.append('%s band, WFD' %f)
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)

    # Median number of visits per field
    metricName = 'NVisits'
    slicerName = 'OpsimFieldSlicer'
    metricMetadata = []
    summaryName = 'Median'
    for f in ('u', 'g', 'r', 'i', 'z', 'y'):
        metricMetadata.append('%s band, WFD' %f)
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)

    # Median coadded depth per field
    metricName = 'CoaddM5'
    slicerName = 'OpsimFieldSlicer'
    metricMetadata = []
    summaryName = 'Median'
    for f in ('u', 'g', 'r', 'i', 'z', 'y'):
        metricMetadata.append('%s band, WFD' %f)
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)

    # fO Nv and A
    metricName = 'fO'
    metricMetadata = 'WFD only (non-dithered)'
    slicerName = None
    summaryName = 'fONv: Area (sqdeg)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)
    summaryName = 'fOArea: Nvisits (#)'
    summary = runCompare.findSummaryStats(metricName, metricMetadata, slicerName, summaryName=summaryName)
    pandaprint(summary)

    # Median r and i band seeing
    metricName = 'Median finSeeing'
    slicerName = None
    summaryName = None
    metricMetadata = []
    for f in (['r', 'i']):
        metricMetadata.append('%s band, WFD' %f)
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)

    # Median ury band sky brightness
    metricName = 'Median filtSkyBrightness'
    slicerName = 'UniSlicer'
    metricMetadata = []
    summaryName= None
    for f in (['u', 'r', 'y']):
        metricMetadata.append('%s band, WFD' %f)
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)
    # Median ury band airmass
    metricName = 'Median airmass'
    for md in metricMetadata:
        summary = runCompare.findSummaryStats(metricName, md, slicerName, summaryName=summaryName)
        pandaprint(summary)

    # Median ury band normalized airmass
    # don't calculate this in maf standard output yet

    # Median ury hour angle
    # don't calculate this in maf standard output yet

    # Close access to the results database files.
    runCompare.close()
