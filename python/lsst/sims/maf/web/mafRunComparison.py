from builtins import zip
from builtins import range
from builtins import object
import os
import warnings
import numpy as np
from lsst.sims.maf.db import ResultsDb
from lsst.sims.maf.db import OpsimDatabaseV4
import pandas as pd

__all__ = ['MafRunComparison']

class MafRunComparison(object):
    """
    Class to read multiple results databases, and find requested summary metric comparisons.
    """
    def __init__(self, baseDir, runlist, rundirs=None):
        """
        Set up the runs to compare and opens connections to all resultsDb_sqlite directories under
        baseDir/runlist[1-N] and their subdirectories.
        Expects a directory structure like:
        baseDir -> run1  -> subdirectory1 (e.g. 'scheduler', containing a resultsDb_sqlite.db file)
        ................ -> subdirectoryN
        ....... -> runN -> subdirectoryX

        Parameters
        ----------
        baseDir : str
            The root directory containing all of the underlying runs and their subdirectories.
        runlist : list
            A list of runs to compare.
        rundirs : list
            A list of directories (relative to baseDir) where the runs in runlist reside.
            Optional - if not provided, assumes directories are simply the names of runlist.
        """
        self.baseDir = baseDir
        self.runlist = runlist
        if rundirs is not None:
            self.rundirs = rundirs
        else:
            self.rundirs = runlist
        self._connect_to_results()

    def _connect_to_results(self):
        """
        Open access to all the results database files.
        Sets nested dictionary of results databases:
        .. dictionary[run1][subdirectory1] = resultsDb
        .. dictionary[run1][subdirectoryN] = resultsDb ...
        """
        # Open access to all results database files in any subdirectories under 'runs'.
        self.runresults = {}
        for r, rdir in zip(self.runlist, self.rundirs):
            self.runresults[r] = {}
            if not os.path.isdir(os.path.join(self.baseDir, r)):
                warnings.warn('Warning: could not find a directory containing analysis results at %s'
                              % (os.path.join(self.baseDir, r)))
            else:
                sublist = os.listdir(os.path.join(self.baseDir, r))
                for s in sublist:
                    if os.path.isfile(os.path.join(self.baseDir, r, s, 'resultsDb_sqlite.db')):
                        self.runresults[r][s] = ResultsDb(outDir=os.path.join(self.baseDir, r, s))
        # Remove any runs from runlist which we could not find results databases for.
        for r in self.runlist:
            if len(self.runresults[r]) == 0:
                warnings.warn('Warning: could not find any results databases for run %s'
                              % (os.path.join(self.baseDir, r)))
                self.runlist.remove(r)

    def close(self):
        """
        Close all connections to the results database files.
        """
        self.__del__()

    def __del__(self):
        for r in self.runresults:
            for s in self.runresults[r]:
                self.runresults[r][s].close()

    def findSummaryStats(self, metricName, metricMetadata=None, slicerName=None, summaryName=None):
        """
        Look for summary metric values matching metricName (and optionally metricMetadata, slicerName
        and summaryName) among the results databases for each run.

        Parameters
        ----------
        metricName : str
            The name of the original metric.
        metricMetadata : str
            The metric metadata specifying the metric desired (optional).
        slicerName : str
            The slicer name specifying the metric desired (optional).
        summaryName : str
            The name of the summary statistic desired (optional).

        Results
        -------
        numpy structured array
            A numpy array containing a summarized metric name, and the metric value (or Nan) for each run.
            metricName     run1             run2         ... runN
            <samplename> <summaryValue1> <summaryValue2> ... <summaryValueN>
        """
        summaryValues = {}
        summaryNames = {}
        for r in self.runlist:
            summaryValues[r] = []
            summaryNames[r] = []
            # Note that we may have more than one matching summary metric value per run.
            for s in self.runresults[r]:
                mId = self.runresults[r][s].getMetricId(metricName=metricName, metricMetadata=metricMetadata,
                                                        slicerName=slicerName)
                if len(mId) > 0:
                    # And we may have more than one summary metric value per resultsDb
                    stats = self.runresults[r][s].getSummaryStats(mId, summaryName=summaryName)
                    for i in range(len(stats['summaryName'])):
                        name = stats['summaryName'][i]
                        if name == 'Identity' or name == 'Id' or name == 'Count':
                            name = ''
                        mName = stats['metricName'][i].replace(';', '')
                        mMetadata = stats['metricMetadata'][i].replace(';', '')
                        sName = stats['slicerName'][i].replace(';', '')
                        if sName == 'UniSlicer':
                            sName = ''
                        summaryNames[r] += [' '.join([name, mName, mMetadata, sName]).rstrip(' ').lstrip(' ')]
                        summaryValues[r] += [stats['summaryValue'][i]]
            if len(summaryValues[r]) == 0:
                warnings.warn("Warning: Found no metric results for %s %s %s %s in run %s"
                              % (metricName, metricMetadata, slicerName, summaryName, r))
        # Recompose into a numpy structured array, now we know how much data we have.
        unique_stats = set()
        for r in self.runlist:
            for name in summaryNames[r]:
                unique_stats.add(name)
        dtype = [('statName', np.str_, 1024)]
        for r in self.runlist:
            dtype += [(r, float)]
        dtype = np.dtype(dtype)
        stats = np.zeros(len(unique_stats), dtype)
        for i, statName in enumerate(unique_stats):
            stats[i][0] = statName
            for j, r in enumerate(self.runlist):
                try:
                    sidx = summaryNames[r].index(statName)
                    stats[i][j + 1] = summaryValues[r][sidx]
                except ValueError:
                    stats[i][j + 1] = np.nan
        return stats

    def variedParameters(self, proposalName, parameterNames):
        sqlconstraint = ''
        for param in parameterNames:
            if len(parameterNames) > 1 and parameterNames[-1] != param:
                sqlconstraint += 'paramName like'+' "%'+proposalName+'%'+param+'%"'+ ' or '
            else:
                sqlconstraint += 'paramName like'+' "%'+proposalName+'%'+param+'%"'
        print ('Querying Config parameters with the following query:')
        print (sqlconstraint)

        opsdb = {}
        for r in self.runlist:
            opsdb[r] = OpsimDatabaseV4(os.path.join(r, 'data', r + '.db'))
        parameterValues = np.ndarray(shape=(len(self.runlist), len(parameterNames)))
        for i, r in enumerate(self.runlist):
            val = opsdb[r].query_columns('Config', colnames=['paramValue'],
                                         sqlconstraint=sqlconstraint)

            for c, v in enumerate(val):
                parameterValues[i][c]= ((v[0]))

        results_df = pd.DataFrame(parameterValues)
        results_df.set_index(np.array(self.runlist),inplace=True)
        results_df.columns = parameterNames

        return results_df

    def mkstandardMetricDict(self):
        standardMetricDict = {'Total Visits': ['NVisits', 'All Visits', 'UniSlicer', 'Count'],
                              'Total Eff Time': ['Total effective time of survey',
                              'All Visits', 'UniSlicer', None],
                              'Nights with Observations': ['Nights with observations',
                              'All Visits', 'UniSlicer', '(days)'],
                              'Median NVists Per Night':['NVisits', 'Per night',
                              'OneDSlicer', 'Median'],
                              'Median Open Shutter Fraction': ['OpenShutterFraction',
                              'Per night', 'OneDSlicer', 'Median'],
                              'Median Slew Time': ['Median slewTime', 'All Visits',
                              'UniSlicer', None],
                              'Mean Slew Time': ['Mean slewTime', 'All Visits',
                              'UniSlicer', None],
                              'Meidan Prop. Mo. 20':['Proper Motion 20',None,None,'Median'],
                              'Meidan Prop. Mo. 24':['Proper Motion 24',None,None,'Median'],
                              'Median Parallax 20':['Parallax 20',
                                                    'All Visits (non-dithered)',
                                                    'HealpixSlicer','Median'],
                              'Median Parallax 24':['Parallax 24',
                                                    'All Visits (non-dithered)',
                                                    'HealpixSlicer',
                                                    'Median'],
                              'Median Parallax Coverage 20':['Parallax Coverage 20',
                                                             'All Visits (non-dithered)',
                                                             'HealpixSlicer','Median'],
                              'Median Parallax Coverage 24':['Parallax Coverage 24',
                                                             'All Visits (non-dithered)',
                                                             'HealpixSlicer',
                                                             'Median']}
        #Seeing Metrics
        for f in (['r', 'i']):
            colName = 'Median seeingFwhmEff ' +f+' band'
            metricName = 'Median seeingFwhmEff'
            slicer = 'UniSlicer'
            metadata = '%s band, WFD' % f
            summary = None
            standardMetricDict[colName] = [metricName,metadata,slicer,summary]
        #CoaddM5 metrics
        for f in ('u', 'g', 'r', 'i', 'z', 'y'):
            colName = 'Median CoaddM5 ' +f+' band'
            metricName = 'CoaddM5'
            slicer = 'OpsimFieldSlicer'
            metadata = '%s band, WFD' % f
            summary = 'Median'
            standardMetricDict[colName] = [metricName,metadata,slicer,summary]
        #HA Range metrics
        for f in ('u', 'g', 'r', 'i', 'z', 'y'):
            colName = 'FullRange HA ' +f+' band'
            metricName = 'FullRange HA'
            slicer = 'OpsimFieldSlicer'
            metadata = '%s band, WFD' % f
            summary = 'Median'
            standardMetricDict[colName] = [metricName,metadata,slicer,summary]

        return standardMetricDict


    def combineSummaryStats(self,proposalName,parameterNames,metricDict=None):

        if metricDict == None:
            metricDict = self.mkstandardMetricDict()
        else:
            metricDict = metricDict

        parameterDataframe = self.variedParameters(proposalName,parameterNames)
        metricVals = {}
        metricDataframe = [None]*len(metricDict)
        for k,m in enumerate(metricDict):
            metricVals[m] = self.findSummaryStats(metricDict[m][0], metricMetadata=metricDict[m][1],
                                                  slicerName=metricDict[m][2], summaryName=metricDict[m][3])
            temp_df = pd.DataFrame(np.vstack(metricVals[m][0])[1:],index=None,dtype=float)
            temp_df.set_index(np.array(self.runlist),inplace=True)
            temp_df.columns = [m]
            metricDataframe[k] = temp_df
        final_df = pd.concat(metricDataframe, axis=1, join_axes=[metricDataframe[0].index])
        final_df = final_df.sort_index(axis=1)
        summaryStatsdf = pd.concat([parameterDataframe,final_df], axis=1,
                                   join_axes=[parameterDataframe.index])
        summaryStatsdf = summaryStatsdf.astype(float)
        return summaryStatsdf,metricDict
