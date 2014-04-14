import numpy as np 
import os
from mafConfig import MafConfig, config2dict, readMetricConfig, readBinnerConfig, readPlotConfig
import warnings
warnings.simplefilter("ignore", Warning) # Suppress tons of numpy warnings

with warnings.catch_warnings() as w:
    warnings.simplefilter("ignore", UserWarning) # Ignore db warning
    import lsst.sims.operations.maf.db as db

import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics
import lsst.sims.operations.maf.utils as utils


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
                name,params,kwargs,plotDict,summaryStats = readMetricConfig(metric)
                kwargs['plotParams'] = plotDict
                # If just one parameter, look up units
                if (len(params) == 1):
                    info = utils.ColInfo()
                    plotDict['_unit'] = info.getUnits(params[0])
                temp_metric = getattr(metrics,metric.name)(*params, **kwargs)
                temp_metric.summaryStats = summaryStats
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
                    gm.setMetrics(self.metricList[binner.index])
                    comment = constr.replace('=','').replace('filter','').replace("'",'').replace('"', '').replace('  ',' ') + binner.metadata
                    gm.runBins(self.data, simDataName=opsimName, metadata=binner.metadata, comment=comment)
                    gm.reduceAll()
                    # Replace the plotParams for selected metricNames
                    for mName in binner.plotConfigs:
                        gm.plotParams[mName] = readPlotConfig(binner.plotConfigs[mName])
                    gm.plotAll(outDir=self.config.outputDir, savefig=True, closefig=True)
                    # Loop through the metrics and calc any summary statistics
                    #import pdb ; pdb.set_trace()
                    for i,metric in enumerate(self.metricList[binner.index]):
                        if hasattr(metric, 'summaryStats'):
                            for stat in metric.summaryStats:
                                if metric.metricDtype == 'object':
                                    baseName = gm.metricNames[i]
                                    all_names = gm.metricValues.keys()
                                    matching_metrics = [x for x in all_names if x[:len(baseName)] == baseName and x != baseName]
                                    for mm in matching_metrics:
                                        summary = gm.computeSummaryStatistics(mm, getattr(metrics,stat))
                                        if type(summary).__name__ == 'float' or type(summary).__name__ == 'int':
                                            summary = np.array(summary)
                                        summary_stats.append(opsimName+','+binner.binnertype+','+constr+','+mm +','+stat+','+ np.array_str(summary))
                                else:
                                    summary = gm.computeSummaryStatistics(metric.name, getattr(metrics,stat))
                                    summary_stats.append(opsimName+','+binner.binnertype+','+constr+','+ metric.name +','+stat+','+ np.array_str(summary))
                    gm.writeAll(outDir=self.config.outputDir)
                    gm.returnOutputFiles(verbose=True)
        f = open(self.config.outputDir+'/summaryStats.dat','w')
        for stat in summary_stats:
            print >>f, stat
        f.close()
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
