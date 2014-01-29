import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.binners as binners
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.binMetrics as binMetrics


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
            temp_binner = getattr(binners,binner.binner)(*binner.params, **binner.kwargs )
            temp_binner.spatialKey1 = binner.spatialKey1
            temp_binner.spatialKey2 = binner.spatialKey2
            temp_binner.leafsize = binner.leafsize
            self.binList.append(temp_binner)
            sub_metricList=[]
            for i,metric in binner.metricDict.iteritems():
                sub_metricList.append(getattr(metrics,metric.metric)
                                       (*metric.params, **metric.kwargs) )
            self.metricList.append(sub_metricList)

        self.constraints = self.config.constraints
        
    def _binKey(self,binner):
        """Take a binner and return the correct type of binMetric"""
        if binner.binnertype == "UNI":
            result = binMetrics.BaseBinMetric()
        elif binner.binnertype == "SPATIAL":
            result = binMetrics.BaseBinMetric()
        return result
    
    def getData(self, tableName,constraint, colnames=[], groupBy='expmjd'):
        """Pull required data from DB """
        #XXX-temporary kludge. Need to decide how to make this intelligent.
        dbTable = tableName 
        table = db.Table(dbTable, 'obsHistID', self.config.dbAddress)
        self.data = table.query_columns_RecArray(constraint=constraint, colnames=colnames, groupByCol=groupBy)
        return 

    def run(self):
        """Loop over each binner and calc metrics for that binner. """
        for opsimName in self.config.opsimNames:
            for j,constr in enumerate(self.constraints):
                for i,binner in enumerate(self.binList):
                    colnames = []
                    for m in self.metricList[i]:
                        for cn in m.colNameList:
                            colnames.append(cn)
                    if binner.binnertype == 'SPATIAL': 
                        colnames.append(binner.spatialKey1) 
                        colnames.append(binner.spatialKey2)
                    colnames = list(set(colnames))
                    self.getData(opsimName,constr, colnames=colnames)
                    #need to add a bit here to calc any needed post-processing columns (e.g., astrometry)
                    gm = self._binKey(binner)
                    if binner.binnertype == 'SPATIAL':
                        binner.setupBinner(self.data[binner.spatialKey1],
                                       self.data[binner.spatialKey2], leafsize=binner.leafsize)
                    if binner.binnertype == 'UNI':
                        binner.setupBinner(self.data)
                    gm.setBinner(binner)
                    gm.runBins(self.metricList[i], self.data, simDataName=opsimName+'%i'%j, metadata='')
                    gm.reduceAll()
                    gm.plotAll(outDir=self.config.outputDir, savefig=True)
                    gm.writeAll(outDir=self.config.outputDir)
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
                    
if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    driver = MafDriver(configOverrideFilename=configOverrideFilename)
    driver.run()





    



 
