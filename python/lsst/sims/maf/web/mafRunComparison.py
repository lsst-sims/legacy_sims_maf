from builtins import zip
from builtins import range
from builtins import object
import os
import warnings
import numpy as np
from lsst.sims.maf.db import ResultsDb
from lsst.sims.maf.db import OpsimDatabaseV4
import pandas as pd
import re
import matplotlib.pyplot as plt

__all__ = ['MafRunComparison']


class MafRunComparison(object):
    """
    Class to read multiple results databases, and find requested summary metric comparisons.
    """
    def __init__(self, baseDir, runlist, rundirs=None, verbose=False):
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
        self.verbose = verbose
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

    def variedParameters(self, paramNameLike=None):
        """
        Query the opsim configuration table for a set of user defined
        configuration parameters for a set of runs.

        Parameters
        ----------
        paramNameLike : list, opt
            A list of of opsim configuration parameters to pull out of the
            configuration table.
        Results
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the configuration
            parameters given in paramName like. The resulting dataframe is
            indexed the name of the opsim runs.
            runName      parameter1         parameter2
            <run_123>   <parameterValues1>  <parameterValues1>

        Notes
        -----
        This method will use the sqlite 'like' function to query the
        configuration table. Below is an example of how the items in
        paramNameLike need to be formatted:
        ["%WideFastDeep%hour_angle_bonus%", "%WideFastDeep%airmass_bonus%"].
        """
        sqlconstraintDict = {}
        parameterNames = [None] * len(paramNameLike)
        if paramNameLike:
            paramNameLike = paramNameLike
        else:
            paramNameLike = ["%WideFastDeep%airmass_bonus%",
                             "%WideFastDeep%hour_angle_bonus%"]
        for i, param in enumerate(paramNameLike):
            starts = [m.start() for m in re.finditer('%', param)]
            parameterNames[i] = param[starts[-2] + 1:starts[-1]]
            sqlconstraintDict[param[starts[-2] + 1:starts[-1]]] = 'paramName like ' + '"' + param + '"'

        opsdb = {}
        for r in self.runlist:
            opsdb[r] = OpsimDatabaseV4(os.path.join(r, 'data', r + '.db'))
        parameterValues = np.ndarray(shape=(len(self.runlist), len(parameterNames)))
        for i, r in enumerate(self.runlist):
            for j, p in enumerate(sqlconstraintDict.keys()):
                val = opsdb[r].query_columns('Config', colnames=['paramValue'],
                                             sqlconstraint=sqlconstraintDict[p])

                if len(val) > 1.0:
                    warnings.warn(sqlconstraintDict[p] + 'returned more than one value.' +
                                  ' add additional information such as the proposal name' +
                                  '\n' + 'Example: ' + '%WideFastDeep%' + p + '%')
                    parameterValues[i][j] = -666
                else:
                    parameterValues[i][j] = val[0][0]
                if self.verbose:
                    print('Queried Config Parameters with: ' + sqlconstraintDict[p] +
                          '\n' + 'found value: ' + str(parameterValues[i][j]))

        results_df = pd.DataFrame(parameterValues)
        results_df.set_index(np.array(self.runlist), inplace=True)
        results_df.columns = sqlconstraintDict.keys()

        return results_df

    def mkstandardMetricDict(self):
        """
        Create a standard dictionary containing all of the metric information
        needed to query the results database of each opsim run.

        """
        standardMetricDict = {'Total Visits': ['NVisits', 'All Visits', 'UniSlicer', 'Count'],
                              'Total Eff Time': ['Total effective time of survey',
                                                 'All Visits', 'UniSlicer', None],
                              'Nights with Observations': ['Nights with observations',
                                                           'All Visits', 'UniSlicer', '(days)'],
                              'Median NVists Per Night': ['NVisits', 'Per night',
                                                          'OneDSlicer', 'Median'],
                              'Median Open Shutter Fraction': ['OpenShutterFraction',
                                                               'Per night', 'OneDSlicer', 'Median'],
                              'Median Slew Time': ['Median slewTime', 'All Visits',
                                                   'UniSlicer', None],
                              'Mean Slew Time': ['Mean slewTime', 'All Visits',
                                                 'UniSlicer', None],
                              'Meidan Prop. Mo. 20': ['Proper Motion 20', None, None, 'Median'],
                              'Meidan Prop. Mo. 24': ['Proper Motion 24', None, None, 'Median'],
                              'Median Parallax 20': ['Parallax 20',
                                                     'All Visits (non-dithered)',
                                                     'HealpixSlicer', 'Median'],
                              'Median Parallax 24': ['Parallax 24',
                                                     'All Visits (non-dithered)',
                                                     'HealpixSlicer',
                                                     'Median'],
                              'Median Parallax Coverage 20': ['Parallax Coverage 20',
                                                              'All Visits (non-dithered)',
                                                              'HealpixSlicer', 'Median'],
                              'Median Parallax Coverage 24': ['Parallax Coverage 24',
                                                              'All Visits (non-dithered)',
                                                              'HealpixSlicer',
                                                              'Median']}
        # Seeing Metrics
        for f in (['r', 'i']):
            colName = 'Median seeingFwhmEff ' + f + ' band'
            metricName = 'Median seeingFwhmEff'
            slicer = 'UniSlicer'
            metadata = '%s band, WFD' % f
            summary = None
            standardMetricDict[colName] = [metricName, metadata, slicer, summary]
        # CoaddM5 metrics
        for f in ('u', 'g', 'r', 'i', 'z', 'y'):
            colName = 'Median CoaddM5 ' + f + ' band'
            metricName = 'CoaddM5'
            slicer = 'OpsimFieldSlicer'
            metadata = '%s band, WFD' % f
            summary = 'Median'
            standardMetricDict[colName] = [metricName, metadata, slicer, summary]
        # HA Range metrics
        for f in ('u', 'g', 'r', 'i', 'z', 'y'):
            colName = 'FullRange HA ' + f + ' band'
            metricName = 'FullRange HA'
            slicer = 'OpsimFieldSlicer'
            metadata = '%s band, WFD' % f
            summary = 'Median'
            standardMetricDict[colName] = [metricName, metadata, slicer, summary]

        return standardMetricDict

    def combineSummaryStats(self, paramNameLike, metricDict=None):
        """
        Combine the summary statistics of a set of metrics into a pandas
        dataframe that is indexed by the opsim run name.

        Parameters
        ----------
        paramNameLike : list
            A list of of opsim configuration parameters to pull out of the
            configuration table.

        metricDict: dict, opt
            A dictionary of metrics with all of the information needed to query
            a results database. If the user does not provide a dictionary
            the standardMetricDict will be used.
        Results
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the configuration
            parameters given in paramName like and a column for each of the
            dictionary keys in the metricDict. The resulting dataframe is
            indexed the name of the opsim runs.
              index       parameter1          parameter2         metric1         metric2
            <run_123>   <parameterValues1>  <parameterValues1>  <metricValue1>  <metricValue2>
            <run_124>   <parameterValues1>  <parameterValues1>  <metricValue1>  <metricValue2>
        """
        if metricDict is None:
            metricDict = self.mkstandardMetricDict()
        else:
            metricDict = metricDict
        parameterDataframe = self.variedParameters(paramNameLike=paramNameLike)
        metricVals = {}
        metricDataframe = [None] * len(metricDict)
        for k, m in enumerate(metricDict):
            metricVals[m] = self.findSummaryStats(metricDict[m][0], metricMetadata=metricDict[m][1],
                                                  slicerName=metricDict[m][2], summaryName=metricDict[m][3])
            temp_df = pd.DataFrame(np.vstack(metricVals[m][0])[1:], index=None, dtype=float)
            temp_df.set_index(np.array(self.runlist), inplace=True)
            temp_df.columns = [m]
            metricDataframe[k] = temp_df
        final_df = pd.concat(metricDataframe, axis=1, join_axes=[metricDataframe[0].index])
        final_df = final_df.sort_index(axis=1)
        summaryStatsdf = pd.concat([parameterDataframe, final_df], axis=1,
                                   join_axes=[parameterDataframe.index])
        summaryStatsdf = summaryStatsdf.astype(float)

        self.parameterDataframe = parameterDataframe
        self.summaryStatsdf = summaryStatsdf
        self.summaryMetricDict = metricDict

    def nomalizeRun(self, baselineRun):
        """
        Normalize the summary metric values in the dataframe
        resulting from combineSummaryStats based on the values of a single
        baseline run.

        Parameters
        ----------
        baselineRun : str
            The name of the opsim run that will serve as baseline.

        Results
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the configuration
            parameters given in paramNamelike and a column for each of the
            dictionary keys in the metricDict. The resulting dataframe is
            indexed the name of the opsim runs.
              index       parameter1          parameter2         metric1         metric2
            <run_123>   <parameterValues1>  <parameterValues1>  <norm_metricValue1>  <morm_etricValue2>
            <run_124>   <parameterValues1>  <parameterValues1>  <orm_metricValue1>  <orm_metricValue2>

        Notes:
        ------
        The metric values are normalized in the following way:

        norm_metric_value(run) = metric_value(run) - metric_value(baselineRun) / metric_value(baselineRun)
        """
        rundf = self.summaryStatsdf
        nparameters = len(self.parameterDataframe.columns)
        print (nparameters)
        num = rundf.iloc[:, nparameters:] - rundf.iloc[:, nparameters:].loc[baselineRun].values.squeeze()
        denom = rundf.iloc[:, nparameters:].loc[baselineRun].values.squeeze()
        noramlize = num / denom
        noramlizedf = pd.concat([rundf.iloc[:, 0:nparameters], noramlize], axis=1,
                                join_axes=[rundf.iloc[:, 0:nparameters].index])

        self.baselineRun = baselineRun
        self.noramlizeStatsdf = noramlizedf

    def normalizedCompPlot(self, output=None, totalVisits=True):
        """
        Plot the normalized metric values as a function of opsim run.

        output: str, opt
            Name of figure to save to disk. If this is left as None the figure
            is not saved.
        totalVisits: bool
            If True the total number of visits is included in the metrics plotted.
            When comparing runs a very different lengths it is recommended to
            set this flag to False.
        """
        ylabel = '(run - ' + self.baselineRun + ')/' + self.baselineRun
        dataframe = self.noramlizeStatsdf
        magcols = [col for col in dataframe.columns if 'M5' in col]
        HAcols = [col for col in dataframe.columns if 'HA' in col]
        propMocols = [col for col in dataframe.columns if 'Prop. Mo.' in col]
        seeingcols = [col for col in dataframe.columns if 'seeing' in col]
        parallaxCols = [col for col in dataframe.columns if 'Parallax' in col]
        if totalVisits is True:
            othercols = ['Mean Slew Time', 'Median Slew Time', 'Median NVists Per Night',
                         'Median Open Shutter Fraction', 'Nights with Observations',
                         'Total Eff Time', 'Total Visits']
        else:
            othercols = ['Mean Slew Time', 'Median Slew Time',
                         'Median NVists Per Night',
                         'Median Open Shutter Fraction']
        colsets = [othercols, magcols, HAcols, propMocols, parallaxCols, seeingcols]
        fig, axs = plt.subplots(len(colsets), 1, figsize=(8, 33))
        fig.subplots_adjust(hspace=.4)
        axs = axs.ravel()
        for i, c in enumerate(colsets):
            x = np.arange(len(dataframe))
            for metric in dataframe[c].columns:
                axs[i].plot(x, dataframe[metric], marker='.', ms=10, label=metric)
            axs[i].grid(True)
            axs[i].set_ylabel(ylabel)
            lgd = axs[i].legend(loc=(1.02, 0.2), ncol=1)
            plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=90)
            plt.setp(axs[i], xticks=x, xticklabels=[x.strip('') for x in dataframe.index.values])
        if output:
            plt.savefig(output, bbox_extra_artists=(lgd,), bbox_inches='tight')
