import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig
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
            temp_binner.setupKwargs = binner.setupKwargs
            temp_binner.constraints = binner.constraints
            self.binList.append(temp_binner)
            sub_metricList=[]
            for i,metric in binner.metricDict.iteritems():
                name,params,kwargs = config2dict(metric)
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
  
    def getData(self, tableName,constraint, colnames=[], groupBy='expMJD'):
        """Pull required data from DB """
        
        dbTable = tableName 
        table = db.Table(dbTable, 'obsHistID', self.config.dbAddress)

        #new idea.  Also pass in any column stacking configs
        #1) loop through the column list, split into a list to pull from DB and post-process list
        #2) initialize the stacking classes, using the config params if present
        #3) add required columns to the db list
        #4) 
        
        pi_amp = False
        normairmass = False
        if 'normairmass' in colnames:
            normairmass = True
            colnames.remove('normairmass')
            colnames.append('airmass') # Need to add this b/c required by normAMStack
        if 'ra_pi_amp' in colnames:
            pi_amp = True
            colnames.remove('ra_pi_amp')
            colnames.remove('dec_pi_amp')
            # Note that astromStack is currently only using fieldRA and fieldDec positions
        colnames=list(set(colnames))
        self.data = table.query_columns_RecArray(constraint=constraint, colnames=colnames, groupByCol=groupBy)
        if normairmass: 
            self.data = utils.normAMStack(self.data)
        if pi_amp:
            self.data = utils.astromStack(self.data)
        return 

    def run(self):
        """Loop over each binner and calc metrics for that binner. """
        for opsimName in self.config.opsimNames:
            for j, constr in enumerate(self.constraints):
                # Find which binners have a matching constraint XXX-need to update to be matching constraint and stacking params
                matchingBinners=[]
                for b in self.binList:
                    if constr in b.constraints:
                        matchingBinners.append(b)
                for i,binner in enumerate(matchingBinners):
                    colnames = []
                    for m in self.metricList[i]:
                        for cn in m.colNameList:  colnames.append(cn)                            
                    if (binner.binnertype == 'SPATIAL') | (binner.binnertype == 'HEALPIX'): 
                        colnames.append(binner.setupParams[0]) 
                        colnames.append(binner.setupParams[1])
                    if binner.binnertype == 'ONED':
                        colnames.append(binner.setupParams[0])
                    colnames = list(set(colnames)) #unique elements
                    self.getData(opsimName,constr, colnames=colnames)     
                    gm = self._binKey(binner)
                    binner.setupBinner(self.data, *binner.setupParams, **binner.setupKwargs)
                    gm.setBinner(binner)
                    gm.setMetrics(self.metricList[i])
                    gm.runBins(self.data, simDataName=opsimName+'%i'%j, metadata='')
                    gm.reduceAll()
                    gm.plotAll(outDir=self.config.outputDir, savefig=True)
                    gm.writeAll(outDir=self.config.outputDir)
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
                    
if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    driver = MafDriver(configOverrideFilename=configOverrideFilename)
    driver.run()





    



 
