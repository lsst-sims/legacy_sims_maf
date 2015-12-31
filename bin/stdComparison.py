import os
import copy
import numpy as np
from lsst.sims.maf.db import ResultsDb


def printResult(summary):
    for s in summary:
        print s['metricName'], s['metricMetadata'], s['slicerName'], s['summaryName'], s['summaryValue']


def printShortresult(runlist, summary, scale=1, scalelabel=None):
    # Keep a (stationary, for the loop) copy of the remaining summary statistics to be printed
    summaryRemaining = copy.deepcopy(summary)
    # Clean out any empty keys
    lengthDict = {}
    for sr in summaryRemaining:
        lengthDict[sr] = len(summaryRemaining[sr])

    for key in lengthDict:
        if lengthDict[key] == 0:
            del summaryRemaining[key]

    while len(summaryRemaining) > 0:
        # Set our "working copy" to the remaining summary statistics
        summary = copy.deepcopy(summaryRemaining)
        sLine = summaryRemaining[summaryRemaining.keys()[0]][0]
        sLineMetricName = sLine['metricName']
        sLineMetricMetadata = sLine['metricMetadata']
        sLineSummaryName = sLine['summaryName']
        #if sLineSummaryName == 'Identity':
        #    sLineSummaryName = ''
        output = '%s ; %s; %s;' %(sLineMetricName.replace(';', ''), sLineMetricMetadata.replace(';', ''),
                                  sLineSummaryName.replace(';', ''))
        if scalelabel is not None:
            output += ' (%s);' %(scalelabel)
        else:
            output += ' ;'
        for r in runlist:
            if r not in summary:
                output += ' ;'
            else:
                foundStat = False
                slist = summary[r]
                for s in slist:
                    if (s['metricName'] == sLineMetricName) and (s['metricMetadata'] == sLineMetricMetadata):
                        output += ' %f;' %(s['summaryValue']*scale)
                        # Delete this summary statistic from the stats which remain to be printed.
                        summaryRemaining[r].remove(s)
                        if len(summaryRemaining[r]) == 0:
                            del summaryRemaining[r]
                        foundStat = True
                if not foundStat:
                    output += ' ;'
        print output

def findStats(runresults, runlist, metricName, metricMetadata=None, slicerName=None, summaryName=None):
    """
    runresults - a dictionary (of dictionaries) of results database objects - top level of dictionary is one per opsim run,
    second level of dictionary is if there are multiple resultsDb's per opsim run (scheduler + science, for example).
    """
    summary = {}
    for r in runlist:
        summary[r] = []
        for d in runresults[r]:
            mId = runresults[r][d].getMetricId(metricName=metricName, metricMetadata=metricMetadata, slicerName=slicerName)
            if len(mId)>0:
                summary[r] += runresults[r][d].getSummaryStats(mId, summaryName=summaryName)
        if len(summary) == 0:
            print "Found no metric results for %s %s %s %s in run %s" %(metricName, metricMetadata, slicerName, summaryName, r)
    return summary



###


runs = ['kraken_1032','kraken_1028','kraken_1029','kraken_1030','kraken_1025',
        'kraken_1031','enigma_1189','enigma_1257',
        'enigma_1258','enigma_1259','ops2_1094', 'kraken_1033', 'kraken_1034',
        'kraken_1035','kraken_1036', 'kraken_1037','kraken_1038',]


# Open access to all the results database files.
runresults = {}
for r in runs:
    runresults[r] = {}
    for d in (['sched', 'sci']):
        resultsDbfile = os.path.join(os.path.join(r, d), 'resultsDb_sqlite.db')
        if os.path.isfile(resultsDbfile):
            runresults[r][d] = ResultsDb(outDir = os.path.join(r, d))
            #print '# Connected to results database at %s' %(resultsDbfile)


# Get 'overview' statistics.

output = 'metricName; metricMetadata; summaryName; scaleLabel;'
for r in runs:
    output +='%s;' %r
print output

# Total number of visits (in millions)
metricName = 'TotalNVisits'
metricMetadata = 'All Visits'
summaryName = 'Count'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary, scale=1/1000000., scalelabel='Millions')

# Total open shutter time (in megasec)
# Need to add this to MAF

# Percentage of visits for each proposal
metricName = 'NVisits Per Proposal'
summaryName = 'Fraction of total'
summary = findStats(runresults, runs, metricName, summaryName=summaryName)
printShortresult(runs, summary, scale=100., scalelabel='percent')

# Mean Surveying efficiency (??)
metricName = 'Total effective time of survey'
metricMetadata = 'All Visits'
summaryName = '(days)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Median open shutter fraction
metricName = 'OpenShutterFraction'
metricMetadata = 'Per night'
summaryName = 'Median'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Mean and Median number of visits per field
metricName = 'Nvisits'
metricMetadata = []
for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    metricMetadata.append('%s band, all props' %f)
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, summaryName='Mean')
    printShortresult(runs, summary)
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, summaryName='Median')
    printShortresult(runs, summary)

# Median number of visits per night
metricName = 'NVisits'
metricMetadata = 'Per night'
slicerName = 'OneDSlicer'
summaryName = 'Median'
summary = findStats(runresults, runs, metricName, metricMetadata, slicerName, summaryName)
printShortresult(runs, summary)

# Number of nights with observations
metricName = 'Nights with observations'
metricMetadata = 'All Visits'
summaryName = '(days)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Mean slew time
metricName = 'Mean slewTime'
summary = findStats(runresults, runs, metricName)
printShortresult(runs, summary)

# fO NV and Area
metricName = 'fO'
metricMetadata = 'All Visits (non-dithered)'
summaryName = 'fONv: Area (sqdeg)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)
summaryName = 'fOArea: Nvisits (#)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Median r band seeing
metricName = 'Median finSeeing'
metricMetadata = 'r band, all props'
summaryName = 'Identity'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Median r band airmass
metricName = 'Median airmass'
metricMetadata = 'r band, all props'
summaryName = 'Identity'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Median proper motion accuracy @20
metricName = 'Proper Motion 20'
summaryName = 'Median'
summary = findStats(runresults, runs, metricName, summaryName=summaryName)
printShortresult(runs, summary)

# Median proper motion accuracy @24
metricName = 'Proper Motion 24'
summaryName = 'Median'
summary = findStats(runresults, runs, metricName, summaryName=summaryName)
printShortresult(runs, summary)

# WFD performance metrics

# Median single visit depth in ugrizy (all visits)
metricName = 'Median fiveSigmaDepth'
slicerName = 'UniSlicer'
metricMetadata = []
summaryName = 'Identity'
for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    metricMetadata.append('%s band, WFD' %f)
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName=slicerName, summaryName=summaryName)
    printShortresult(runs, summary)

# Median number of visits per field
metricName = 'Nvisits'
slicerName = 'OpsimFieldSlicer'
metricMetadata = []
summaryName = 'Median'
for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    metricMetadata.append('%s band, WFD' %f)
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName=slicerName, summaryName=summaryName)
    printShortresult(runs, summary)

# Median coadded depth per field
metricName = 'CoaddM5'
slicerName = 'OpsimFieldSlicer'
metricMetadata = []
summaryName = 'Median'
for f in ('u', 'g', 'r', 'i', 'z', 'y'):
    metricMetadata.append('%s band, WFD' %f)
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName=slicerName, summaryName=summaryName)
    printShortresult(runs, summary)

# fO Nv and A
metricName = 'fO'
metricMetadata = 'WFD only (non-dithered)'
summaryName = 'fONv: Area (sqdeg)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)
summaryName = 'fOArea: Nvisits (#)'
summary = findStats(runresults, runs, metricName, metricMetadata, summaryName=summaryName)
printShortresult(runs, summary)

# Median r and i band seeing
metricName = 'Median finSeeing'
metricMetadata = []
for f in (['r', 'i']):
    metricMetadata.append('%s band, WFD' %f)
slicerName = 'UniSlicer'
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName)
    printShortresult(runs, summary)

# Median ury band sky brightness
metricName = 'Median filtSkyBrightness'
metricMetadata = []
for f in (['u', 'r', 'y']):
    metricMetadata.append('%s band, WFD' %f)
slicerName = 'UniSlicer'
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName)
    printShortresult(runs, summary)

# Median ury band airmass
metricName = 'Median airmass'
for md in metricMetadata:
    summary = findStats(runresults, runs, metricName, md, slicerName)
    printShortresult(runs, summary)

# Median ury band normalized airmass
# don't calculate this in maf standard output yet

# Median ury hour angle
# don't calculate this in maf standard output yet

# Close access to the results database files.
for r in runs:
    for d in runresults[r]:
        #print '# Closing %s %s' %(r, d)
        runresults[r][d].close()
