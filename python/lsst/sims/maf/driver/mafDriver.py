import numpy as np 
import os
from .mafConfig import MafConfig, config2dict, readMetricConfig, readBinnerConfig, readPlotConfig
import warnings
warnings.simplefilter("ignore", Warning) # Suppress tons of numpy warnings

with warnings.catch_warnings() as w:
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.maf.db as db

import lsst.sims.maf.binners as binners
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.binMetrics as binMetrics
import lsst.sims.maf.utils as utils
import time

class MafDriver(object):
    """Script for configuring and running metrics on Opsim output """

    def __init__(self, configOverrideFilename=None):
        """Load up the configuration and set the bin and metric lists """
        self.config=MafConfig()
        # Load any config file
        if configOverrideFilename is not None:
            self.config.load(configOverrideFilename)

        # Load any parameters set on the command line

        # Validate and freeze the config
        self.config.validate()
        self.config.freeze()

        # Check for output directory, make if needed
        if not os.path.isdir(self.config.outputDir):
            os.makedirs(self.config.outputDir)

        # Construct the binners and metric objects
        self.binList = []
        self.metricList = []
        for i,binner in self.config.binners.iteritems():
            name, params, kwargs, setupParams,setupKwargs, metricDict, constraints, stackCols,plotDict,metadata = readBinnerConfig(binner)
            temp_binner = getattr(binners,binner.name)(*params, **kwargs )
            temp_binner.setupParams = setupParams
            temp_binner.setupKwargs = setupKwargs
            temp_binner.constraints = binner.constraints
            #check that constraints in binner are unique
            if len(temp_binner.constraints) > len(set(temp_binner.constraints)):
                print 'Binner %s has repeated constraints'%binner.name
                print 'Constraints:  ', binner.constraints
                raise Exception('Binner constraints are not unique')
            temp_binner.plotConfigs = binner.plotConfigs
            temp_binner.metadata = metadata
            temp_binner.index = i
            temp_binner.binnertype = temp_binner.binnerName[:4].upper() # Matching baseBinMetric
            stackers = []
            for key in stackCols.keys():
                name, params, kwargs = config2dict(stackCols[key])
                stackers.append(getattr(utils.addCols, name)(*params, **kwargs))
            temp_binner.stackers = stackers
            self.binList.append(temp_binner)
            sub_metricList=[]
            for j,metric in binner.metricDict.iteritems():
                name,params,kwargs,plotDict,summaryStats,histMerge = readMetricConfig(metric) # Need to make summaryStats a dict with keys of metric names and items of kwarg dicts.
                kwargs['plotParams'] = plotDict
                # If just one parameter, look up units
                if (len(params) == 1):
                    info = utils.ColInfo()
                    plotDict['_units'] = info.getUnits(params[0])
                temp_metric = getattr(metrics,metric.name)(*params, **kwargs)
                temp_metric.summaryStats = []
                for key in summaryStats.keys():
                    temp_metric.summaryStats.append(getattr(metrics,key)('metricdata',**readPlotConfig(summaryStats[key])))
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
        for i,binner in enumerate(self.binList):
            for constraint in binner.constraints:
                for metric in self.metricList[i]:
                    # Approximate what output filename will be 
                    comment = constraint.replace('=','').replace('filter','').replace("'",'').replace('"', '').replace('  ',' ') + binner.metadata
                    filenames.append('_'.join([metric.name, comment, binner.binnertype]))
        if len(filenames) != len(set(filenames)):
            duplicates = list(set([x for x in filenames if filenames.count(x) > 1]))
            counts = [filenames.count(x) for x in duplicates]
            print ['%s: %d versions' %(d, c) for d, c in zip(duplicates, counts)]
            raise Exception('Filenames for metrics will not be unique.  Add binner metadata or change metric names.')
        
  
    def getData(self, tableName,constraint, colnames=[], stackers=[], groupBy='expMJD'):
        """Pull required data from DB """
        
        dbTable = tableName 
        table = db.Table(dbTable, 'obsHistID', self.config.dbAddress['dbAddress'])

        stacker_names = [s.name for s in stackers ]
        dbcolnames = []
        sourceLookup = utils.getColInfo.ColInfo()
        for name in colnames:
            source = sourceLookup.getDataSource(name)
            if source:
                for col in source.cols:  dbcolnames.append(col)
                # If we don't have a configured stacker, make a default one
                if source.name not in stacker_names: 
                    stackers.append(source)
                    stacker_names.append(source.name)
            else:
                dbcolnames.append(name)
        # If we need stackers, make sure they get columns they need
        for stacker in stackers:
            for col in stacker.cols:
                dbcolnames.append(col)
        dbcolnames=list(set(dbcolnames))
        self.data = table.query_columns_RecArray(constraint=constraint, colnames=dbcolnames, groupByCol=groupBy)

        for stacker in stackers:
            self.data = stacker.run(self.data)
            


    def getFieldData(self, binner):
        """Given an opsim binner, generate the FieldData """
        if 'fieldTable' in self.config.dbAddress.keys():
            if not hasattr(self, 'fieldData'): # Only pull the data once if getting it from the database
                fieldDataInfo = self.config.dbAddress
                self.fieldData = utils.getData.fetchFieldsFromFieldTable(fieldDataInfo['fieldTable'],
                                                                fieldDataInfo['dbAddress'],
                                                                sessionID=fieldDataInfo['sessionID'],
                                                                proposalTable=fieldDataInfo['proposalTable'],
                                                                proposalID=fieldDataInfo['proposalID'])
        else:
            fieldID, idx = np.unique(self.data[binner.simDataFieldIdColName], return_index=True)
            ra = self.data[binner.fieldRaColName][idx]
            dec = self.data[binner.fieldDecColName][idx]
            self.fieldData = np.core.records.fromarrays([fieldID, ra, dec],
                                               names=['fieldID', 'fieldRA', 'fieldDec'])
     
            
    
    def run(self):
        """Loop over each binner and calc metrics for that binner. """
        summary_stats=[]
        summary_stats.append('opsimname,binnertype,sql where, metric name, summary stat name, value')
        for opsimName in self.config.opsimNames:
            for j, constr in enumerate(self.constraints):
                # Find which binners have a matching constraint 
                matchingBinners=[]
                binnertypes=[]
                for b in self.binList:
                    if constr in b.constraints:
                        matchingBinners.append(b)
                        binnertypes.append(b.binnertype)
                colnames=[]
                for i,binner in enumerate(matchingBinners):
                    for m in self.metricList[binner.index]:
                        for cn in m.colNameList:  colnames.append(cn)
                    for cn in binner.columnsNeeded:
                        colnames.append(cn)
                    for stacker in binner.stackers:
                        for col in stacker.cols:
                            colnames.append(col)
                colnames = list(set(colnames)) #unique elements
                    
                print 'Running SQLconstraint:', constr
                self.getData(opsimName, constr, colnames=colnames)
                if len(self.data) == 0:
                    print 'No data matching constraint:   %s'%constr
                else:
                    if 'OPSI' in binnertypes:
                        self.getFieldData(matchingBinners[binnertypes.index('OPSI')])
                    # so maybe here pool.apply_async(runBinMetric, constriant=const, colnames=colnames, binners=matchingBinners, metricList=self.metricList, dbAdress=self.config.dbAddress, outdir=self.config.outputDir)
                    for i,binner in enumerate(matchingBinners):
                        # Thinking about how to run in parallel...I think this loop would be a good place (although there wouldn't be any speedup for querries that only use one binner...If we run the getData's in parallel, run the risk of hammering the database and/or running out of memory. Maybe run things in parallel inside the binMetric? 
                        # what could I do--write a function that takes:  simdata, binners, metriclist, dbAdress.
                        # could use the config file to set how many processors to use in the pool.
                        print '  with binnertype =', binner.binnertype, 'metrics:', ', '.join([m.name for m in self.metricList[binner.index]])
                        for stacker in binner.stackers:
                            self.data = stacker.run(self.data)
                        gm = binMetrics.BaseBinMetric() 
                        if binner.binnertype == 'OPSI':
                            # Need to pass in fieldData as well
                            binner.setupBinner(self.data, self.fieldData,*binner.setupParams, **binner.setupKwargs )
                        else:
                            binner.setupBinner(self.data, *binner.setupParams, **binner.setupKwargs)
                        gm.setBinner(binner)
                        metricNames_in_gm = gm.setMetrics(self.metricList[binner.index])
                        comment = constr.replace('=','').replace('filter','').replace("'",'').replace('"', '').replace('  ',' ') + binner.metadata
                        gm.runBins(self.data, simDataName=opsimName, metadata=binner.metadata, comment=comment)
                        gm.reduceAll()
                        # Replace the plotParams for selected metricNames
                        for mName in binner.plotConfigs:
                            gm.plotParams[mName] = readPlotConfig(binner.plotConfigs[mName])
                        gm.plotAll(outDir=self.config.outputDir, savefig=True, closefig=True)
                        # Loop through the metrics and calc any summary statistics
                        for i,metric in enumerate(self.metricList[binner.index]):
                            if hasattr(metric, 'summaryStats'):
                                for stat in metric.summaryStats:
                                    # If it's a complex metric, run summary stats on each reduced metric
                                    if metric.metricDtype == 'object':
                                        baseName = gm.metricNames[i]
                                        all_names = gm.metricValues.keys()
                                        matching_metrics = [x for x in all_names if x[:len(baseName)] == baseName and x != baseName]
                                        for mm in matching_metrics:
                                            summary = gm.computeSummaryStatistics(mm, stat)
                                            if type(summary).__name__ == 'float' or type(summary).__name__ == 'int':
                                                summary = np.array(summary)
                                            summary_stats.append(opsimName+','+binner.binnertype+','+constr+','+mm +','+stat.name+','+ np.array_str(summary))
                                    else:
                                        summary = gm.computeSummaryStatistics(metric.name, stat)
                                        summary_stats.append(opsimName+','+binner.binnertype+','+constr+','+ metric.name +','+stat.name+','+ np.array_str(summary))
                        gm.writeAll(outDir=self.config.outputDir)
                        # Return Output Files - get file output key back. Verbose=True, prints to screen.
                        outFiles = gm.returnOutputFiles(verbose=False)
                        # Loop through the outFiles and attach them to the correct metric in self.metricList.  This would probably be better with a dict.
                        outfile_names = []
                        outfile_metricNames = []
                        for outfile in outFiles:
                            if outfile['filename'][-3:] == 'npz':
                                outfile_names.append(outfile['filename'])
                                outfile_metricNames.append(outfile['metricName'])
                        for i,m in enumerate(self.metricList[binner.index]):
                            good = np.where(np.array(outfile_metricNames) == metricNames_in_gm[i])[0]
                            m.saveFile = outfile_names[good]
                            
        f = open(self.config.outputDir+'/summaryStats.dat','w')
        for stat in summary_stats:
            print >>f, stat
        f.close()
        # Merge any histograms that need merging.  While doing a write/read is not efficient, it will make it easier to convert the big loop above to parallel later.  
        
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
                        if hasattr(m,'saveFile') and key in histDict.keys():  # Could be there was no data, then it got skipped
                            histDict[key]['files'].append(m.saveFile)
                            temp_dict = m.histMerge
                            del temp_dict['histNum']
                            histDict[key]['plotkwargs'].append(temp_dict)

        
        for key in histDict.keys():
            cbm = binMetrics.ComparisonBinMetric()
            if len(histDict[key]['files']) > 0:
                for filename in histDict[key]['files']:
                    cbm.readMetricData(filename)
                dictNums = cbm.binmetrics.keys()
                dictNums.sort()
                cbm.plotHistograms(dictNums,[cbm.binmetrics[0].metricNames[0]]*len(dictNums),
                                   outDir=self.config.outputDir, savefig=True,
                                   plotkwargs=histDict[key]['plotkwargs'])

        today_date, versionInfo = utils.getDateVersion()
        # Open up a file and print the results of verison and date.
        datefile = open(self.config.outputDir+'/'+'date_version_ran.dat','w')
        print >>datefile, 'date, version, fingerprint '
        #import pdb ; pdb.set_trace()
        print >>datefile, '%s,%s,%s'%(today_date,versionInfo['__version__'],versionInfo['__fingerprint__'])
        datefile.close()
        # Save the as-ran pexConfig file
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
        

