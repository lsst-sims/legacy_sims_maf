import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig, config2dict, readMetricConfig, readBinnerConfig, readPlotConfig
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
            temp_binner.index=i
            stackers = []
            for key in stackCols.keys():
                name, params, kwargs = config2dict(stackCols[key])
                stackers.append(getattr(utils.addCols, name)(*params, **kwargs))
            temp_binner.stackers = stackers
            self.binList.append(temp_binner)
            sub_metricList=[]
            for j,metric in binner.metricDict.iteritems():
                name,params,kwargs,plotDict = readMetricConfig(metric)
                kwargs['plotParams'] = plotDict
                # If just one parameter, look up units
                if (len(params) == 1):
                    info = utils.ColInfo()
                    plotDict['_unit'] = info.getUnits(params[0])
                sub_metricList.append(getattr(metrics,metric.name)
                                      (*params, **kwargs) )
            self.metricList.append(sub_metricList)
        # Make a unique list of all SQL constraints
        #import pdb ; pdb.set_trace()
        self.constraints = []
        for b in self.binList:
            for c in b.constraints:
                self.constraints.append(c)
        self.constraints = list(set(self.constraints))
        
        
    def _binKey(self,binner): #XXX-looks like BaseBinMetric does everything.  Should we just call it BinnerMetricContainer?
        """Take a binner and return the correct type of binMetric"""
        if binner.binnertype == "UNI":
            result = binMetrics.BaseBinMetric()
        elif (binner.binnertype == "SPATIAL") | (binner.binnertype == "HEALPIX") :
            result = binMetrics.BaseBinMetric()
        else:
            result = binMetrics.BaseBinMetric()
        return result
  
    def getData(self, tableName,constraint, colnames=[], stackers=[], groupBy='expMJD'):
        """Pull required data from DB """
        
        dbTable = tableName 
        table = db.Table(dbTable, 'obsHistID', self.config.dbAddress)

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
        dbcolnames=list(set(dbcolnames))
        self.data = table.query_columns_RecArray(constraint=constraint, colnames=dbcolnames, groupByCol=groupBy)

        for stacker in stackers:
            self.data = stacker.run(self.data)
            
        
    def run(self):
        """Loop over each binner and calc metrics for that binner. """
        for opsimName in self.config.opsimNames:
            for j, constr in enumerate(self.constraints):
                # Find which binners have a matching constraint 
                matchingBinners=[]
                for b in self.binList:
                    if constr in b.constraints:
                        matchingBinners.append(b)
                for i,binner in enumerate(matchingBinners):
                    print 'running constraint:', constr,' with binnertype =', binner.binnertype 
                    colnames = []
                    for m in self.metricList[binner.index]:
                        for cn in m.colNameList:  colnames.append(cn)
                    for cn in binner.columnsNeeded:
                        colnames.append(cn)
                    colnames = list(set(colnames)) #unique elements
                    self.getData(opsimName,constr, colnames=colnames, stackers=binner.stackers)     
                    gm = self._binKey(binner)
                    #import pdb ; pdb.set_trace()
                    binner.setupBinner(self.data, *binner.setupParams, **binner.setupKwargs)
                    gm.setBinner(binner)
                    gm.setMetrics(self.metricList[binner.index])
                    gm.runBins(self.data, simDataName=opsimName+'%i'%j, metadata=binner.metadata)
                    gm.reduceAll()
                    # Replace the plotParams for selected metricNames
                    for mName in binner.plotConfigs:
                        gm.plotParams[mName] = readPlotConfig(binner.plotConfigs[mName])
                    gm.plotAll(outDir=self.config.outputDir, savefig=True, outfileRoot=constr)
                    gm.writeAll(outDir=self.config.outputDir, outfileRoot=constr)
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
                    
if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    driver = MafDriver(configOverrideFilename=configOverrideFilename)
    driver.run()





    



 
