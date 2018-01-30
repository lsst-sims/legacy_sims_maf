from builtins import zip
from builtins import range
from builtins import object
import os
import warnings
import numpy as np
import pandas as pd
from lsst.sims.maf.db import ResultsDb
from lsst.sims.maf.db import OpsimDatabase
import lsst.sims.maf.metricBundles as mb
import lsst.sims.maf.plots as plots

__all__ = ['RunComparison']


class RunComparison(object):
    """
    Class to read multiple results databases, find requested summary metric comparisons,
    and stores results in DataFrames in class.

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
        Optional - if not provided, assumes directories are simply the names in runlist.
        Must have same length as runlist (note that runlist can contain duplicate entries).
    """
    def __init__(self, baseDir, runlist, rundirs=None,
                 defaultResultsDb='resultsDb_sqlite.db', verbose=False):
        self.baseDir = baseDir
        self.runlist = runlist
        self.verbose = verbose
        self.defaultResultsDb = defaultResultsDb
        if rundirs is not None:
            if len(rundirs) != len(runlist):
                raise ValueError('runlist and rundirs must be the same length')
            self.rundirs = rundirs
        else:
            self.rundirs = runlist
        self._connect_to_results()
        # Class attributes to store the stats data:
        self.parameters = None        # Config parameters varied in each run
        self.headerStats = None       # Save information on the summary stat values
        self.summaryStats = None      # summary stats
        self.normalizedStats = None   # normalized (to baselineRun) version of the summary stats
        self.baselineRun = None       # name of the baseline run

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
            checkdir = os.path.join(self.baseDir, rdir)
            if not os.path.isdir(checkdir):
                warnings.warn('Warning: could not find a directory at %s' % checkdir)
            else:
                # Add a dictionary to runresults to store resultsDB connections.
                if r not in self.runresults:
                    self.runresults[r] = {}
                # Check for a resultsDB in the current checkdir
                if os.path.isfile(os.path.join(checkdir, self.defaultResultsDb)):
                    s = os.path.split(rdir)[-1]
                    self.runresults[r][s] = ResultsDb(outDir=checkdir)
                # And look for resultsDb files in subdirectories.
                sublist = os.listdir(checkdir)
                for s in sublist:
                    if os.path.isfile(os.path.join(checkdir, s, 'resultsDb_sqlite.db')):
                        self.runresults[r][s] = ResultsDb(outDir=os.path.join(checkdir, s))
        # Remove any runs from runlist which we could not find results databases for.
        for r in self.runlist:
            if len(self.runresults[r]) == 0:
                warnings.warn('Warning: could not find any results databases for run %s'
                              % (os.path.join(self.baseDir, r)))
        self.runlist = list(self.runresults.keys())

    def close(self):
        """
        Close all connections to the results database files.
        """
        self.__del__()

    def __del__(self):
        for r in self.runresults:
            for s in self.runresults[r]:
                self.runresults[r][s].close()

    def variedParameters(self, paramNameLike=None, dbDir=None):
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
        if paramNameLike is None:
            paramNameLike = ["%WideFastDeep%airmass_bonus%",
                             "%WideFastDeep%hour_angle_bonus%"]
        sqlconstraints = []
        parameterNames = []
        for p in paramNameLike:
            name = p.rstrip('%').lstrip('%').replace('%', ' ')
            parameterNames.append(name)
            sql = 'paramName like "%s"' % (p)
            sqlconstraints.append(sql)

        # Connect to original databases and grab configuration parameters.
        opsdb = {}
        for r in self.runlist:
            # Check if file exists XXXX
            if dbDir is None:
                opsdb[r] = OpsimDatabase(os.path.join(r, 'data', r + '.db'))
            else:
                opsdb[r] = OpsimDatabase(os.path.join(dbDir, r + '.db'))
                # try also sqlite.db
        parameterValues = {}
        for i, r in enumerate(self.runlist):
            parameterValues[r] = {}
            for pName, sql in zip(parameterNames, sqlconstraints):
                val = opsdb[r].query_columns('Config', colnames=['paramValue'],
                                             sqlconstraint=sql)
                if len(val) > 1.0:
                    warnings.warn(sql + ' returned more than one value.' +
                                  ' Add additional information such as the proposal name' +
                                  '\n' + 'Example: ' + '%WideFastDeep%hour_angle_bonus%')
                    parameterValues[r][pName] = -666
                else:
                    parameterValues[r][pName] = val['paramValue'][0]
                if self.verbose:
                    print('Queried Config Parameters with: ' + sql +
                          '\n' + 'found value: ' + str(parameterValues[r][pName]))
        tempDFList = []
        for r in self.runlist:
            tempDFList.append(pd.DataFrame(parameterValues[r], index=[r]))
        # Concatenate dataframes for each run.
        if self.parameters is None:
            self.parameters = pd.concat(tempDFList)
        else:
            self.parameters = self.parameters.join(tempDFList)

    def buildMetricDict(self, metricNameLike=None, metricMetadataLike=None,
                       slicerNameLike=None, subdir=None):
        """Return a metric dictionary based on finding all metrics which match 'like' the various parameters.

        Parameters
        ----------
        metricNameLike: str, opt
            Metric name like this -- i.e. will look for metrics which match metricName like "value".
        metricMetadataLike: str, opt
            Metric Metadata like this.
        slicerNameLike: str, opt
            Slicer name like this.
        subdir: str, opt
            Find metrics from this subdir only.
            If other parameters are not specified, this returns all metrics within this subdir.

        Returns
        -------
        Dict
            Key = self-created metric 'name', value = Dict{metricName, metricMetadata, slicerName}
        """
        if metricNameLike is None and metricMetadataLike is None and slicerNameLike is None:
            getAll = True
        else:
            getAll = False
        mDict = {}
        for r in self.runlist:
            if subdir is not None:
                subdirs = [subdir]
            else:
                subdirs = list(self.runresults[r].keys())
            for subdir in subdirs:
                if getAll:
                    mIds = self.runresults[r][subdir].getAllMetricIds()
                else:
                    mIds = self.runresults[r][subdir].getMetricIdLike(metricNameLike=metricNameLike,
                                                                      metricMetadataLike=metricMetadataLike,
                                                                      slicerNameLike=slicerNameLike)
                for mId in mIds:
                    info = self.runresults[r][subdir].getMetricDisplayInfo(mId)
                    metricName = info['metricName'][0]
                    metricMetadata = info['metricMetadata'][0]
                    slicerName = info['slicerName'][0]
                    name = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
                    mDict[name] = {'metricName': metricName,
                                   'metricMetadata': metricMetadata,
                                   'slicerName': slicerName}
        return mDict

    def _buildSummaryName(self, metricName, metricMetadata, slicerName, summaryStatName):
        if metricMetadata is None:
            metricMetadata = ''
        if slicerName is None:
            slicerName = ''
        sName = summaryStatName
        if sName == 'Identity' or sName == 'Id' or sName == 'Count' or sName is None:
            sName = ''
        slName = slicerName
        if slName == 'UniSlicer':
            slName = ''
        name = ' '.join([sName, metricName, metricMetadata, slName]).rstrip(' ').lstrip(' ')
        name.replace(',', '')
        return name

    def _findSummaryStats(self, metricName, metricMetadata=None, slicerName=None, summaryName=None,
                          colName=None):
        """
        Look for summary metric values matching metricName (and optionally metricMetadata, slicerName
        and summaryName) among the results databases for each run.

        Parameters
        ----------
        metricName : str
            The name of the original metric.
        metricMetadata : str, opt
            The metric metadata specifying the metric desired (optional).
        slicerName : str, opt
            The slicer name specifying the metric desired (optional).
        summaryName : str, opt
            The name of the summary statistic desired (optional).
        colName : str, opt
            Name of the column header for the dataframe. If more than one summary stat is
            returned from the database, then this will be ignored.

        Results
        -------
        Pandas Dataframe
            <index>   <metricName>  (possibly additional metricNames - multiple summary stats or metadata..)
             runName    value
        """
        summaryValues = {}
        summaryNames = {}
        for r in self.runlist:
            summaryValues[r] = {}
            summaryNames[r] = {}
            # Check if this metric/metadata/slicer/summary stat name combo is in
            # this resultsDb .. or potentially in another subdirectory's resultsDb.
            for subdir in self.runresults[r]:
                mId = self.runresults[r][subdir].getMetricId(metricName=metricName,
                                                             metricMetadata=metricMetadata,
                                                             slicerName=slicerName)
                # Note that we may have more than one matching summary metric value per run.
                if len(mId) > 0:
                    # And we may have more than one summary metric value per resultsDb
                    stats = self.runresults[r][subdir].getSummaryStats(mId, summaryName=summaryName)
                    if len(stats['summaryName']) == 1 and colName is not None:
                        name = colName
                        summaryValues[r][name] = stats['summaryValue'][0]
                        summaryNames[r][name] = stats['summaryName'][0]
                    else:
                        for i in range(len(stats['summaryName'])):
                            name = self._buildSummaryName(metricName, metricMetadata, slicerName,
                                                          stats['summaryName'][i])
                            summaryValues[r][name] = stats['summaryValue'][i]
                            summaryNames[r][name] = stats['summaryName'][i]
            if len(summaryValues[r]) == 0:
                warnings.warn("Warning: Found no metric results for %s %s %s %s in run %s"
                              % (metricName, metricMetadata, slicerName, summaryName, r))
        # Make DataFrame.
        # First determine list of all summary stats pulled from all databases.
        unique_stats = set()
        for r in self.runlist:
            for name in summaryNames[r]:
                unique_stats.add(name)
        # Make sure every runName (key) in summaryValues dictionary has a value for each stat.
        for s in unique_stats:
            for r in self.runlist:
                try:
                    summaryValues[r][s]
                except KeyError:
                    summaryValues[r][s] = np.nan
        # Create data frames for each run. This is the simplest way to handle it in pandas.
        summaryBase = {}
        mName = {}
        mData = {}
        sName = {}
        basemetricname = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
        for s in unique_stats:
            summaryBase[s] = basemetricname
            mName[s] = metricName
            mData[s] = metricMetadata
            sName[s] = slicerName
        tempDFHeader = [pd.DataFrame(summaryBase, index=['BaseName'])]
        tempDFHeader.append(pd.DataFrame(mName, index=['MetricName']))
        tempDFHeader.append(pd.DataFrame(mData, index=['Metadata']))
        tempDFHeader.append(pd.DataFrame(sName, index=['Slicer']))
        tempDFHeader.append(pd.DataFrame(summaryNames[r], index=['SummaryType']))
        header = pd.concat(tempDFHeader)
        tempDFList = []
        for r in self.runlist:
            tempDFList.append(pd.DataFrame(summaryValues[r], index=[r]))
        # Concatenate dataframes for each run.
        stats = pd.concat(tempDFList)
        return header, stats

    def addSummaryStats(self, metricDict):
        """
        Combine the summary statistics of a set of metrics into a pandas
        dataframe that is indexed by the opsim run name.

        Parameters
        ----------
        metricDict: dict
            A dictionary of metrics with all of the information needed to query
            a results database.  The metric/metadata/slicer/summary values referred to
            by a metricDict value could be unique but don't have to be.

        Returns
        -------
        pandas DataFrame
            A pandas dataframe containing a column for each of the configuration
            parameters given in paramName like and a column for each of the
            dictionary keys in the metricDict. The resulting dataframe is
            indexed the name of the opsim runs.
              index      metric1         metric2
            <run_123>    <metricValue1>  <metricValue2>
            <run_124>    <metricValue1>  <metricValue2>
        """
        for mName, metric in metricDict.items():
            if 'summaryName' not in metric:
                metric['summaryName'] = None
                tempHeader, tempStats = self._findSummaryStats(metricName=metric['metricName'],
                                                               metricMetadata=metric['metricMetadata'],
                                                               slicerName=metric['slicerName'],
                                                               summaryName=metric['summaryName'],
                                                               colName=mName)
            if self.summaryStats is None:
                self.summaryStats = tempStats
                self.headerStats = tempHeader
            else:
                self.summaryStats = self.summaryStats.join(tempStats, lsuffix='_x')
                self.headerStats = self.headerStats.join(tempHeader, lsuffix='_x')

    def normalizeStats(self, baselineRun):
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
            index        metric1               metric2
            <run_123>    <norm_metricValue1>  <norm_metricValue2>
            <run_124>    <norm_metricValue1>  <norm_metricValue2>

        Notes:
        ------
        The metric values are normalized in the following way:

        norm_metric_value(run) = metric_value(run) - metric_value(baselineRun) / metric_value(baselineRun)
        """
        self.normalizedStats = self.summaryStats.copy(deep=True)
        self.normalizedStats = self.normalizedStats - self.summaryStats.loc[baselineRun]
        self.normalizedStats /= self.summaryStats.loc[baselineRun]
        self.baselineRun = baselineRun

    def sortCols(self, baseName=True, summaryType=True):
        """Return the columns (in order) to display a sorted version of the stats dataframe.

        Parameters
        ----------
        baseName : bool, opt
            Sort by the baseName. Default True.
            If True, this takes priority in the sorted results.
        summaryType : bool, opt
            Sort by the summary stat name (summaryType). Default True.

        Returns
        -------
        list
        """
        sortby = []
        if baseName:
            sortby.append('BaseName')
        if summaryType:
            sortby.append('SummaryType')
        o = self.headerStats.sort_values(by=sortby, axis=1)
        return o.columns

    def filterCols(self, summaryType):
        """Return a dataframe containing only stats which match summaryType.

        Parameters
        ----------
        summaryType : str
            The type of summary stat to match. (i.e. Max, Mean)

        Returns
        -------
        pd.DataFrame
        """
        o = self.headerStats.loc['SummaryType'] == summaryType
        return self.summaryStats.loc[:, o]

    def findChanges(self, threshold=0.05):
        """Return a dataframe containing only values which changed by threshhold.

        Parameters
        ----------
        threshold : float, opt
            Identify values which change by more than threshold (%) in the normalized values.
            Default 5% (0.05).

        Returns
        -------
        pd.DataFrame
        """
        o = abs(self.normalizedStats) > 0.05
        o = o.any(axis=0)
        return self.summaryStats.loc[:, o]

    def getFileNames(self, metricName, metricMetadata=None, slicerName=None):
        """For each of the runs in runlist, get the paths to the datafiles for a given metric.

        Parameters
        ----------
        metricName : str
            The name of the original metric.
        metricMetadata : str, opt
            The metric metadata specifying the metric desired (optional).
        slicerName : str, opt
            The slicer name specifying the metric desired (optional).

        Returns
        -------
        Dict
            Keys: runName, Value: path to file
        """
        filepaths = {}
        for r in self.runlist:
            for s in self.runresults[r]:
                mId = self.runresults[r][s].getMetricId(metricName=metricName,
                                                        metricMetadata=metricMetadata,
                                                        slicerName=slicerName)
                if len(mId) > 0 :
                    if len(mId) > 1:
                        warnings.warn("Found more than one metric data file matching " +
                                      "metricName %s metricMetadata %s and slicerName %s"
                                      % (metricName, metricMetadata, slicerName) +
                                      " Skipping this combination.")
                    else:
                        filename = self.runresults[r][s].getMetricDataFiles(metricId=mId)
                        filepaths[r] = os.path.join(r, s, filename[0])
        return filepaths

    def plotSummaryStats(self):
        # We'll fix this asap
        pass

    # Plot actual metric values (skymaps or histograms or power spectra) (values not stored in class).
    def readMetricData(self, metricName, metricMetadata, slicerName):
        # Get the names of the individual files for all runs.
        # Dictionary, keyed by run name.
        filenames = self.getFileNames(metricName, metricMetadata, slicerName)
        mname = self._buildSummaryName(metricName, metricMetadata, slicerName, None)
        bundleDict = {}
        for r in filenames:
            bundleDict[r] = mb.createEmptyMetricBundle()
            bundleDict[r].read(filenames[r])
        return bundleDict, mname

    def _parameterTitles(self, run, paramCols=None):
        tempDict = self.parameters.loc[run].to_dict()
        tempTitle = run
        if paramCols is None:
            paramCols = self.parameters.columns
        for col in paramCols:
            tempTitle = tempTitle  + '\n' + col + ' ' + str(tempDict[col])
        return tempTitle


    def plotMetricData(self, bundleDict, plotFunc, runlist=None, userPlotDict=None,
                       layout=None, outDir=None, paramTitles=False, paramCols=None, savefig=False):
        if runlist is None:
            runlist = self.runlist
        if userPlotDict is None:
            userPlotDict = {}

        ph = plots.PlotHandler(outDir=outDir, savefig=savefig)
        bundleList = []
        for r in runlist:
            bundleList.append(bundleDict[r])
        ph.setMetricBundles(bundleList)

        plotDicts = [{} for r in runlist]
        # Depending on plotFunc, overplot or make many subplots.
        if plotFunc.plotType == 'SkyMap':
            # Note that we can only handle 9 subplots currently due
            # to how subplot identification (with string) is handled.
            if len(runlist) > 9:
                raise ValueError('Please try again with < 9 subplots for skymap.')
            # Many subplots.
            if 'colorMin' not in userPlotDict:
                colorMin = 100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().min()
                        colorMin = min(tmp, colorMin)
                    else:
                        colorMin = bundleDict[b].plotDict['colorMin']
                userPlotDict['colorMin'] = colorMin
            if 'colorMax' not in userPlotDict:
                colorMax = -100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().max()
                        colorMax = max(tmp, colorMax)
                    else:
                        colorMax = bundleDict[b].plotDict['colorMax']
                userPlotDict['colorMax'] = colorMax
            for i, pdict in enumerate(plotDicts):
                # Add user provided dictionary.
                pdict.update(userPlotDict)
                # Set subplot information.
                if layout is None:
                    ncols = int(np.ceil(np.sqrt(len(runlist))))
                    nrows = int(np.ceil(len(runlist) / float(ncols)))
                else:
                    ncols = layout[0]
                    nrows = layout[1]
                pdict['subplot'] = int(str(nrows) + str(ncols) + str(i + 1))
                if paramTitles is False:
                    pdict['title'] = runlist[i]
                else:
                    pdict['title'] = self._parameterTitles(runlist[i], paramCols=paramCols)
                # For the subplots we do not need the label
                pdict['label'] = ''
                if 'suptitle' not in userPlotDict:
                    pdict['suptitle'] = ph._buildTitle()
        elif plotFunc.plotType == 'Histogram':
            # Put everything on one plot.
            if 'xMin' not in userPlotDict:
                xMin = 100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().min()
                        xMin = min(tmp, xMin)
                    else:
                        xMin = bundleDict[b].plotDict['xMin']
                userPlotDict['xMin'] = xMin
            if 'xMax' not in userPlotDict:
                xMax = -100000000
                for b in bundleDict:
                    if 'zp' not in bundleDict[b].plotDict:
                        tmp = bundleDict[b].metricValues.compressed().max()
                        xMax = max(tmp, xMax)
                    else:
                        xMax = bundleDict[b].plotDict['xMax']
                userPlotDict['xMax'] = xMax
            for i, pdict in enumerate(plotDicts):
                pdict.update(userPlotDict)
                pdict['subplot'] = '111'
                # Legend and title will automatically be ok, I think.
        elif plotFunc.plotType == 'BinnedData':
            # Put everything on one plot.
            if 'yMin' not in userPlotDict:
                yMin = 100000000
                for b in bundleDict:
                    tmp = bundleDict[b].metricValues.compressed().min()
                    yMin = min(tmp, yMin)
                userPlotDict['yMin'] = yMin
            if 'yMax' not in userPlotDict:
                yMax = -100000000
                for b in bundleDict:
                    tmp = bundleDict[b].metricValues.compressed().max()
                    yMax = max(tmp, yMax)
                userPlotDict['yMax'] = yMax
            if 'xMin' not in userPlotDict:
                xMin = 100000000
                for b in bundleDict:
                    tmp = bundleDict[b].slicer.slicePoints['bins'].min()
                    xMin = min(tmp, xMin)
                userPlotDict['xMin'] = xMin
            if 'xMax' not in userPlotDict:
                xMax = -100000000
                for b in bundleDict:
                    tmp = bundleDict[b].slicer.slicePoints['bins'].max()
                    xMax = max(tmp, xMax)
                userPlotDict['xMax'] = xMax
            for i, pdict in enumerate(plotDicts):
                pdict.update(userPlotDict)
                pdict['subplot'] = '111'
                # Legend and title will automatically be ok, I think.
        if self.verbose:
            print(plotDicts)
        ph.plot(plotFunc, plotDicts=plotDicts)
