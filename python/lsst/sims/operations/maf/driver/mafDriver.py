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
        binconfigs = [self.config.bin1,self.config.bin2,self.config.bin3,
                   self.config.bin4,self.config.bin5,
                   self.config.bin6,self.config.bin7,
                   self.config.bin8,self.config.bin9,self.config.bin10]
        s_bins = []
        s_binKwrds = []
        s_metrics =[]
        s_metricParams =[]
        s_metricKwrds =[]
        self.constraints = []
        for b in binconfigs:
            s_bins.append(b.binner)
            s_binKwrds.append(b.kwrds)
            s_metrics.append(b.metrics)
            s_metricParams.append(b.metricParams)
            s_metricKwrds.append(b.metricKwrds)
            self.constraints.append(b.constraints) 
            
        self.binList=[]
        self.metricList=[]

        for i,s_bin in enumerate(s_bins):
            if s_bin is not '':
                self.binList.append(getattr(binners,s_bin)(**eval('dict('+ s_binKwrds[i]+')'))   )
                sub_metricList=[]
                for j,s_metric in enumerate(s_metrics[i]):
                    sub_metricList.append(getattr(metrics,s_metric)(*s_metricParams[i][j].split(','), **eval('dict('+s_metricKwrds[i][j]+')' )))
                self.metricList.append(sub_metricList)


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
        for i,binner in enumerate(self.binList):
            for opsimName in self.config.opsimNames:
                for j,constr in enumerate(self.constraints[i]): #If I have different binners w/same cosntraints, this is fetching same data multiple times.  Could do something more clever and only pull once.  This seems like a nice spot to multiprocess the sucker too.
                    colnames = []
                    for m in self.metricList[i]:
                        for cn in m.colNameList:
                            colnames.append(cn)
                    if binner.binnertype == 'SPATIAL': 
                        colnames.append(self.config.spatialKey1) 
                        colnames.append(self.config.spatialKey2)
                    colnames = list(set(colnames))
                    self.getData(opsimName,constr, colnames=colnames)
                    #need to add a bit here to calc any needed post-processing columns (e.g., astrometry)
                    gm = self._binKey(binner)
                    if binner.binnertype == 'SPATIAL':
                        binner.setupBinner(self.data[self.config.spatialKey1],
                                       self.data[self.config.spatialKey2], leafsize=self.config.leafsize)
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





    



 
