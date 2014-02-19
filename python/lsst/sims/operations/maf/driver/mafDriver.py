import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig, config2dict, readMetricConfig
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
            temp_binner = getattr(binners,binner.name)(*binner.params, **binner.kwargs )
            temp_binner.setupParams = binner.setupParams
            temp_binner.setupKwargs = {}
            for key in binner.setupKwargs_float.keys():
                temp_binner.setupKwargs[key] =  binner.setupKwargs_float[key]
            for key in binner.setupKwargs_str.keys():
                temp_binner.setupKwargs[key] =  binner.setupKwargs_str[key]
            temp_binner.constraints = binner.constraints
            temp_binner.index=i
            stackers = []
            for key in binner.stackCols.keys():
                name, params, kwargs = config2dict(binner.stackCols[key])
                stackers.append(getattr(utils.addCols, name)(*params, **kwargs))
            temp_binner.stackers = stackers
            self.binList.append(temp_binner)
            sub_metricList=[]
            for j,metric in binner.metricDict.iteritems():
                name,params,kwargs,plotDict = readMetricConfig(metric)
                kwargs['plotParams'] = plotDict
                sub_metricList.append(getattr(metrics,metric.name)
                                      (*params, **kwargs) )
            self.metricList.append(sub_metricList)
        # Make a unique list of all SQL constraints
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
                    #print 'constraint = ', constr,'binnertype =', binner.binnertype 
                    colnames = []
                    for m in self.metricList[binner.index]:
                        for cn in m.colNameList:  colnames.append(cn)                            
                    if (binner.binnertype == 'SPATIAL') | (binner.binnertype == 'HEALPIX'): 
                        colnames.append(binner.setupParams[0]) 
                        colnames.append(binner.setupParams[1])
                    if binner.binnertype == 'ONED':
                        colnames.append(binner.setupParams[0])
                    if binner.binnertype == 'OPSIMFIELDS':
                        colnames.append(binner.setupParams[0])
                        colnames.append(binner.setupParams[1])
                        colnames.append(binner.setupParams[2])
                    colnames = list(set(colnames)) #unique elements
                    self.getData(opsimName,constr, colnames=colnames, stackers=binner.stackers)     
                    gm = self._binKey(binner)
                    binner.setupBinner(self.data, *binner.setupParams, **binner.setupKwargs)
                    gm.setBinner(binner)
                    gm.setMetrics(self.metricList[binner.index])
                    gm.runBins(self.data, simDataName=opsimName+'%i'%j)
                    gm.reduceAll()
                    gm.plotAll(outDir=self.config.outputDir, savefig=True, outfileRoot=constr)
                    gm.writeAll(outDir=self.config.outputDir, outfileRoot=constr)
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
                    
if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    driver = MafDriver(configOverrideFilename=configOverrideFilename)
    driver.run()





    



 
