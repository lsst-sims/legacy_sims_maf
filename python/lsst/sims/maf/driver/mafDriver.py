import os
import warnings
import numpy as np 

from .mafConfig import MafConfig, config2dict, readMetricConfig, readSlicerConfig, readMixConfig
warnings.simplefilter("ignore", Warning) # Suppress tons of numpy warnings

import lsst.sims.maf.db as db
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.sliceMetrics as sliceMetrics
import lsst.sims.maf.utils as utils
import time


def dtime(time_prev):
   return (time.time() - time_prev, time.time())



class MafDriver(object):
    """Script for configuring and running metrics on Opsim output """

    def __init__(self, configvalues):
        """Load up the configuration and set the bin and metric lists """
        # Configvalues passed from runDriver.py
        self.config = configvalues

        # Validate and freeze the config
        self.config.validate()
        self.config.freeze()

        # Check for output directory, make if needed.
        if not os.path.isdir(self.config.outputDir):
            os.makedirs(self.config.outputDir)

        self.verbose = self.config.verbose
        self.figformat = self.config.figformat
        self.dpi = self.config.dpi
            
        # Set up database connection.
        self.opsimdb = utils.connectOpsimDb(self.config.dbAddress)

        time_prev = time.time()
        # Grab config info and write to disk.
        if self.config.getConfig:
            configSummary, configDetails = self.opsimdb.fetchConfig()
            f = open(os.path.join(self.config.outputDir,'configSummary.txt'), 'w')
            utils.outputUtils.printDict(configSummary, 'Config Summary', filehandle=f)
            f.close()
            f = open(os.path.join(self.config.outputDir, 'configDetails.txt'), 'w')
            utils.outputUtils.printDict(configDetails, 'Config Details', filehandle=f)
            f.close()
            if self.verbose:
                dt, time_prev = dtime(time_prev)
                print 'Got OpSim config info in %.3g s'%dt

        self.allpropids, self.wfdpropids, self.ddpropids = self.opsimdb.fetchPropIDs()
        if self.verbose:
            dt, time_prev = dtime(time_prev)
            print 'fetched PropID info in %.3g s'%dt
        # Construct the slicers and metric objects
        self.binList = []
        self.metricList = []
        for i,slicer in self.config.slicers.iteritems():
            name, params, kwargs, setupParams, setupKwargs, metricDict, constraints, stackCols, plotDict, metadata = readSlicerConfig(slicer)
            temp_slicer = getattr(slicers,slicer.name)(*params, **kwargs )
            temp_slicer.setupParams = setupParams
            temp_slicer.setupKwargs = setupKwargs
            temp_slicer.constraints = slicer.constraints
            #check that constraints in slicer are unique
            if len(temp_slicer.constraints) > len(set(temp_slicer.constraints)):
                print 'Slicer %s has repeated constraints'%slicer.name
                print 'Constraints:  ', slicer.constraints
                raise Exception('Slicer constraints are not unique')
            temp_slicer.plotConfigs = slicer.plotConfigs
            temp_slicer.metadata = metadata
            temp_slicer.index = i
            stackers = []
            for key in stackCols.keys():
                name, params, kwargs = config2dict(stackCols[key])
                stackers.append(getattr(utils.addCols, name)(*params, **kwargs))
            temp_slicer.stackers = stackers
            self.binList.append(temp_slicer)
            sub_metricList=[]
            for metric in slicer.metricDict.itervalues():
                name,params,kwargs,plotDict,summaryStats,histMerge = readMetricConfig(metric)
                # Need to make summaryStats a dict with keys of metric names and items of kwarg dicts.
                kwargs['plotParams'] = plotDict
                # If just one parameter, look up units
                if (len(params) == 1):
                    info = utils.ColInfo()
                    plotDict['_units'] = info.getUnits(params[0])
                temp_metric = getattr(metrics,metric.name)(*params, **kwargs)
                temp_metric.summaryStats = []
                for key in summaryStats.keys():
                    temp_metric.summaryStats.append(getattr(metrics,key)('metricdata',**readMixConfig(summaryStats[key])))
                # If it is a UniSlicer, make sure the IdentityMetric is run
                if temp_slicer.slicerName == 'UniSlicer':
                   if 'IdentityMetric' not in summaryStats.keys():
                      temp_metric.summaryStats.append(getattr(metrics,'IdentityMetric')('metricdata'))
                temp_metric.histMerge = histMerge
                sub_metricList.append(temp_metric )
            self.metricList.append(sub_metricList)
        # Make a unique list of all SQL constraints
        self.constraints = []
        for b in self.binList:
            for c in b.constraints:
                self.constraints.append(c)
        self.constraints = list(set(self.constraints))
        # Check that all filenames will be unique
        filenames=[]
        for i,slicer in enumerate(self.binList):
            for constraint in slicer.constraints:
                for metric in self.metricList[i]:
                    # Approximate what output filename will be 
                    comment = constraint.replace('=','').replace('filter','').replace("'",'')
                    comment = comment.replace('"', '').replace('  ',' ') + slicer.metadata
                    filenames.append('_'.join([metric.name, comment, slicer.slicerName]))
        if len(filenames) != len(set(filenames)):
            duplicates = list(set([x for x in filenames if filenames.count(x) > 1]))
            counts = [filenames.count(x) for x in duplicates]
            print ['%s: %d versions' %(d, c) for d, c in zip(duplicates, counts)]
            raise Exception('Filenames for metrics will not be unique.  Add slicer metadata or change metric names.')
  
    def getData(self, constraint, colnames=[], stackers=[], groupBy='expMJD'):
        """Pull required data from database and calculate additional columns from stackers. """
        # Stacker_names describe the already-configured (via the config driver) stacker methods.
        stacker_names = [s.__class__.__name__ for s in stackers ]
        dbcolnames = []
        sourceLookup = utils.getColInfo.ColInfo()
        # Go through all columns that the metrics need.
        for colname in colnames:
            source = sourceLookup.getDataSource(colname)
            # If data source of column is a stacker:
            if source != sourceLookup.defaultDataSource:
                source = getattr(utils.addCols, source)()
                for col in source.colsReq:
                    # Add column names that the stackers need.
                    dbcolnames.append(col)
                # If not already a configured stacker, instantiate one using defaults
                if source.__class__.__name__ not in stacker_names: 
                    stackers.append(source)
                    stacker_names.append(source.__class__.__name__)
            # Else if data source is just the usual database:
            else:
                dbcolnames.append(colname)
        # Remove duplicates from list of columns required from database.
        dbcolnames=list(set(dbcolnames))
        # Get the data from database.
        self.data = self.opsimdb.fetchMetricData(sqlconstraint=constraint, colnames=dbcolnames, groupBy = groupBy)
        # Calculate the data from stackers.
        for stacker in stackers:
            self.data = stacker.run(self.data)
        # Done - self.data should now have all required columns.
            

    def getFieldData(self, slicer, sqlconstraint):
        """Given an opsim slicer, generate the FieldData """
        # Do a bunch of parsing to get the propids out of the sqlconstraint.
        if 'propID' not in sqlconstraint:
            propids = self.allpropids
        else:
            # example sqlconstraint: filter = r and (propid = 219 or propid = 155) and propid!= 90
            sqlconstraint = sqlconstraint.replace('=', ' = ').replace('(', '').replace(')', '')
            sqlconstraint = sqlconstraint.replace("'", '').replace('"', '')
            # Allow for choosing all but a particular proposal.
            sqlconstraint = sqlconstraint.replace('! =' , ' !=')
            sqlconstraint = sqlconstraint.replace('  ', ' ')
            sqllist = sqlconstraint.split(' ')
            propids = []
            nonpropids = []
            i = 0
            while i < len(sqllist):
                if sqllist[i].lower() == 'propid':
                    i += 1
                    if sqllist[i] == "=":
                        i += 1
                        propids.append(int(sqllist[i]))
                    elif sqllist[i] == '!=':
                        i += 1
                        nonpropids.append(int(sqllist[i]))
                i += 1
            if len(propids) == 0:
                propids = self.allpropids
            if len(nonpropids) > 0:
                for nonpropid in nonpropids:
                    if nonpropid in propids:
                        propids.remove(nonpropid)
        # And query the field Table.
        if 'fieldTable' in self.opsimdb.tables:
            self.fieldData = self.opsimdb.fetchFieldsFromFieldTable(propids)
        else:
            fieldID, idx = np.unique(self.data[slicer.simDataFieldIdColName], return_index=True)
            ra = self.data[slicer.fieldRaColName][idx]
            dec = self.data[slicer.fieldDecColName][idx]
            self.fieldData = np.core.records.fromarrays([fieldID, ra, dec],
                                               names=['fieldID', 'fieldRA', 'fieldDec'])
     
            
    
    def run(self):
        """Loop over each slicer and calculate metrics for that slicer. """
        
        # Start a list to hold the output file names.
        allOutfiles = []
        allOutDict = {}
        
        # Start up summary stats running commentary.
        summary_stats=[]
        # Add header to summary stats.
        summary_stats.append('##opsimname,slicer_name,sql_where,metric_name,summary_stat_name,value')

        # Loop through all sqlconstraints, and run slicers + metrics that match the same sql constraints
        #   (so we only have to do one query of database per sql constraint).
        for sqlconstraint in self.constraints:
            # Find which slicers have an exactly matching constraint
            matchingSlicers=[]
            slicerNames=[]
            for b in self.binList:
                if sqlconstraint in b.constraints:
                    matchingSlicers.append(b)
                    slicerNames.append(b.slicerName)
            # And for those slicers, find the data columns required.
            colnames=[]
            stackers = []
            for slicer in matchingSlicers:
                for m in self.metricList[slicer.index]:
                    for cn in m.colNameList:
                        colnames.append(cn)
                for cn in slicer.columnsNeeded:
                    colnames.append(cn)
                for stacker in slicer.stackers:
                    stackers.append(stacker)
                    for col in stacker.colsReq:
                        colnames.append(col)
            # Find the unique column names required.
            colnames = list(set(colnames)) 
            
            print 'Querying with SQLconstraint:', sqlconstraint
            # Get the data from the database + stacker calculations.
            if self.verbose:
                time_prev = time.time()
            self.getData(sqlconstraint, colnames=colnames, stackers=stackers)
            if self.verbose:
                dt, time_prev = dtime(time_prev)
            if len(self.data) == 0:
                print '  No data matching constraint:   %s'%sqlconstraint
                
            # Got data, now set up slicers.
            else:
                if self.verbose:
                    print '  Found %i matching visits in %.3g s'%(len(self.data),dt)
                else:
                    print '  Found %i matching visits'%len(self.data)
                # Special data requirements for opsim slicer.
                if 'OpsimFieldSlicer' in slicerNames:
                    self.getFieldData(matchingSlicers[slicerNames.index('OpsimFieldSlicer')], sqlconstraint)
                # Setup each slicer, and run through the binpoints (with metrics) in baseSliceMetric
                if self.verbose:
                    time_prev = time.time()
                for slicer in matchingSlicers:
                    print '    running slicerName =', slicer.slicerName, \
                      ' run metrics:', ', '.join([m.name for m in self.metricList[slicer.index]])
                    # Set up slicer.
                    if slicer.slicerName == 'OpsimFieldSlicer':
                        # Need to pass in fieldData as well
                        slicer.setupSlicer(self.data, self.fieldData, *slicer.setupParams, **slicer.setupKwargs )
                    else:
                        slicer.setupSlicer(self.data, *slicer.setupParams, **slicer.setupKwargs)
                    # Set up baseSliceMetric.
                    gm = sliceMetrics.BaseSliceMetric(figformat=self.figformat, dpi=self.dpi) 
                    gm.setSlicer(slicer)
                    metricNames_in_gm = gm.setMetrics(self.metricList[slicer.index])
                    # Make a more useful metadata comment.
                    metadata = sqlconstraint.replace('=','').replace('filter','').replace("'",'')
                    metadata = metadata.replace('"', '').replace('  ',' ') + slicer.metadata
                    # Run through binpoints in slicer, and calculate metric values.
                    gm.runSlices(self.data, simDataName=self.config.opsimName,
                               metadata=metadata, sqlconstraint=sqlconstraint)
                    if self.verbose:
                       dt,time_prev = dtime(time_prev)
                       print '    Computed metrics in %.3g s'%dt
                    # And run reduce methods for relevant metrics.
                    gm.reduceAll()
                    # Replace the plotParams for selected metricNames (to allow override from config file).
                    for mName in slicer.plotConfigs:
                        gm.plotParams[mName] = readMixConfig(slicer.plotConfigs[mName])
                    # And plot all metric values.
                    gm.plotAll(outDir=self.config.outputDir, savefig=True, closefig=True, verbose=True)
                    if self.verbose:
                       dt,time_prev = dtime(time_prev)
                       print '    plotted metrics in %.3g s'%dt
                    # Loop through the metrics and calculate any summary statistics
                    for i, metric in enumerate(self.metricList[slicer.index]):
                        if hasattr(metric, 'summaryStats'):
                            for stat in metric.summaryStats:
                                # If it's metric returning an OBJECT, run summary stats on each reduced metric
                                # (have to identify related reduced metric values first)
                                if metric.metricDtype == 'object':
                                    baseName = gm.metricNames[i]
                                    all_names = gm.metricValues.keys()
                                    matching_metrics = [x for x in all_names \
                                                        if x[:len(baseName)] == baseName and x != baseName]
                                    for mm in matching_metrics:
                                        summary = gm.computeSummaryStatistics(mm, stat)
                                        if np.size(summary) == 1:
                                           summary = np.array(np.asscalar(summary))
                                        statstring = self.config.opsimName + ',' + slicer.slicerName + ',' + sqlconstraint 
                                        statstring += ','  + mm + ',' + stat.name + ',' + np.array_str(summary)
                                        summary_stats.append(statstring)
                                # Else it's a simple metric value.
                                else:
                                    summary = gm.computeSummaryStatistics(metric.name, stat)
                                    if np.size(summary) == 1:
                                       summary = np.array(np.asscalar(summary))
                                    statstring = self.config.opsimName + ',' + slicer.slicerName + ',' + sqlconstraint 
                                    statstring += ',' + metric.name + ',' + stat.name + ',' + np.array_str(summary)
                                    summary_stats.append(statstring)
                    if self.verbose:
                       dt,time_prev = dtime(time_prev)
                       print '    Computed summarystats in %.3g s'%dt
                    # And write metric data files to disk.
                    gm.writeAll(outDir=self.config.outputDir)
                    if self.verbose:
                       dt,time_prev = dtime(time_prev)
                       print '    wrote output files in %.3g s'%dt
                    # Grab output Files - get file output key back. Verbose=True, prints to screen.
                    outFiles = gm.returnOutputFiles(verbose=False)
                    # Build continual dictionary of all output info over multiple sqlconstraints.
                    for metrickey in outFiles:
                        hashkey = '%d_%s' %(slicer.index, metrickey)
                        # Shouldn't overwrite previous keys, but let's check.
                        i = 0
                        while hashkey in allOutDict:
                            hashkey = hashkey + '_%d' %(i)
                            i += 1
                        allOutDict[hashkey] = outFiles[metrickey]
                        allOutfiles.append(outFiles[metrickey]['dataFile'])
                    # And keep track of which output files hold metric data (for merging histograms)
                    outfile_names = []
                    outfile_metricNames = []
                    for key in outFiles:
                        outfile_names.append(outFiles[key]['dataFile'])
                        outfile_metricNames.append(outFiles[key]['metricName'])
                    for i,m in enumerate(self.metricList[slicer.index]):
                        good = np.where(np.array(outfile_metricNames) == metricNames_in_gm[i])[0]
                        m.saveFile = outfile_names[good]
                #if self.verbose:
                #    dt,time_prev = dtime(time_prev)
                #    print '    Computed made plots in %.3g s'%dt
        # Save summary statistics to file.
        f = open(self.config.outputDir+'/summaryStats.dat','w')
        for stat in summary_stats:
            print >>f, stat
        f.close()
        
        # Create any 'merge' histograms that need merging.
        # Loop through all the metrics and find which histograms need to be merged
        histList = []
        for m1 in self.metricList:
            for m in m1:
                if 'histNum' in m.histMerge.keys():
                    histList.append(m.histMerge['histNum'])
        
        histList = list(set(histList))
        histList.sort()
        histDict={}
        for item in histList:
            histDict[item] = {}
            histDict[item]['files']=[]
            histDict[item]['plotkwargs']=[]
                        
            for m1 in self.metricList:
                for m in m1:
                    if 'histNum' in m.histMerge.keys():
                        key = m.histMerge['histNum']
                        # Could be there was no data, then it got skipped
                        if hasattr(m,'saveFile') and key in histDict.keys():
                            histDict[key]['files'].append(m.saveFile)
                            temp_dict = m.histMerge
                            del temp_dict['histNum']
                            histDict[key]['plotkwargs'].append(temp_dict)

        
        for key in histDict.keys():
            cbm = sliceMetrics.ComparisonSliceMetric(verbose=False, figformat=self.figformat, dpi=self.dpi)
            if len(histDict[key]['files']) > 0:
                for filename in histDict[key]['files']:
                    fullfilename = os.path.join(self.config.outputDir, filename)
                    cbm.readMetricData(fullfilename)
                dictNums = cbm.slicemetrics.keys()
                dictNums.sort()
                fignum, title, outfile = cbm.plotHistograms(dictNums,[cbm.slicemetrics[0].metricNames[0]]*len(dictNums),
                                                     outDir=self.config.outputDir, savefig=True,
                                                     plotkwargs=histDict[key]['plotkwargs'])
                # Add this plot info to the allOutDict ('ResultsSummary.dat')
                key = 0
                while key in allOutDict:
                    key += 1
                allOutDict[key] = {}
                metricName = cbm.slicemetrics[0].metricNames[0]
                allOutDict[key]['metricName'] = metricName
                allOutDict[key]['simDataName'] = self.config.opsimName
                allOutDict[key]['slicerName'] = cbm.slicemetrics[0].slicer.slicerName
                allOutDict[key]['metadata'] = title
                allOutDict[key]['sqlconstraint'] = ''
                allOutDict[key]['comboPlot'] = outfile
                
        # Save metric filekey & summary stats output. 
        summaryfile = open(os.path.join(self.config.outputDir, 'ResultsSummary.dat'), 'w')
        subkeyorder = ['metricName', 'simDataName', 'slicerName', 'metadata', 'sqlconstraint', 'dataFile']
        utils.outputUtils.printSimpleDict(allOutDict, subkeyorder, summaryfile, delimiter=', ')
        summaryfile.close()
                
        today_date, versionInfo = utils.getDateVersion()
        # Open up a file and print the results of verison and date.
        datefile = open(self.config.outputDir+'/'+'date_version_ran.dat','w')
        print >>datefile, 'date, version, fingerprint '
        print >>datefile, '%s,%s,%s'%(today_date,versionInfo['__version__'],versionInfo['__fingerprint__'])
        datefile.close()
        # Save the list of output files
        np.save(self.config.outputDir+'/'+'outputFiles.npy', allOutfiles)
        # Save the as-ran pexConfig file
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
        

