import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig
import lsst.sims.operations.maf.db as db
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics
import lsst.sims.operations.maf.gridMetrics as gridMetrics


class MafDriver(object):
    """Script for configuring and running metrics on Opsim output """

    def __init__(self, configOverrideFilename=None):
        """Load up the configuration and set the grid and metric lists """
        self.config=MafConfig()
        # Load any config file
        if configOverrideFilename is not None:
            self.config.load(configOverrideFilename)

        # Load any parameters set on the command line

        # Validate and freeze the config
        self.config.validate()
        self.config.freeze()

        # Construct the grid and metric objects
        s_grids = [self.config.grid1,self.config.grid2,self.config.grid3,self.config.grid4,self.config.grid5,
                   self.config.grid6,self.config.grid7,self.config.grid8,self.config.grid9,self.config.grid10]
        s_gridKwrds = [self.config.kwrdsForGrid1,self.config.kwrdsForGrid2,self.config.
                       kwrdsForGrid3,self.config.kwrdsForGrid4,self.config.kwrdsForGrid5,
                       self.config.kwrdsForGrid6,self.config.kwrdsForGrid7,self.config.kwrdsForGrid8,
                       self.config.kwrdsForGrid9,self.config.kwrdsForGrid10 ]
        s_metrics = [self.config.metricsForGrid1,self.config.metricsForGrid2, self.config.metricsForGrid3,
                     self.config.metricsForGrid4, self.config.metricsForGrid5, self.config.metricsForGrid6,
                     self.config.metricsForGrid7, self.config.metricsForGrid8, self.config.metricsForGrid9,
                     self.config.metricsForGrid10]
        s_metricParams = [self.config.metricParamsForGrid1,self.config.metricParamsForGrid2
                          ,self.config.metricParamsForGrid3,self.config.metricParamsForGrid4,self.config.metricParamsForGrid5,
                          self.config.metricParamsForGrid6,self.config.metricParamsForGrid7,self.config.metricParamsForGrid8,
                          self.config.metricParamsForGrid9,self.config.metricParamsForGrid10]
        s_metricKwrds = [self.config.metricKwrdsForGrid1,self.config.metricKwrdsForGrid2,self.config.metricKwrdsForGrid3,
                         self.config.metricKwrdsForGrid4,self.config.metricKwrdsForGrid5,
                         self.config.metricKwrdsForGrid6,self.config.metricKwrdsForGrid7,
                         self.config.metricKwrdsForGrid8,self.config.metricKwrdsForGrid9,self.config.metricKwrdsForGrid10]
        self.constraints = [self.config.constraintsForGrid1,self.config.constraintsForGrid2,self.config.constraintsForGrid3,
                       self.config.constraintsForGrid4,self.config.constraintsForGrid5,self.config.constraintsForGrid6,
                       self.config.constraintsForGrid7,self.config.constraintsForGrid8,self.config.constraintsForGrid9,
                       self.config.constraintsForGrid10]

        self.gridList=[]
        self.metricList=[]

        for i,s_grid in enumerate(s_grids):
            if s_grid is not '':
                self.gridList.append(getattr(grids,s_grid)(**eval('dict('+ s_gridKwrds[i]+')'))   )
                sub_metricList=[]
                for j,s_metric in enumerate(s_metrics[i]):
                    sub_metricList.append(getattr(metrics,s_metric)(*s_metricParams[i][j].split(','), **eval('dict('+s_metricKwrds[i][j]+')' )))
                self.metricList.append(sub_metricList)


    def _gridKey(self,grid):
        """Take a grid and return the correct type of gridmetric"""
        if grid.gridtype == "GLOBAL":
            result = gridMetrics.GlobalGridMetric()
        elif grid.gridtype == "SPATIAL":
            result = gridMetrics.SpatialGridMetric()
        return result
    
    def getData(self, tableName,constraint, colnames=[], groupBy='expmjd'):
        """Pull required data from DB """
        #XXX-temporary kludge. Need to decide how to make this intelligent.
        dbTable = tableName 
        table = db.Table(dbTable, 'obsHistID', self.config.dbAddress)
        self.data = table.query_columns_RecArray(constraint=constraint, colnames=colnames, groupByCol=groupBy)
        return 

    def run(self):
        """Loop over each grid and calc metrics for that grid. """
        for i,grid in enumerate(self.gridList):
            for opsimName in self.config.opsimNames:
                for j,constr in enumerate(self.constraints[i]): #If I have different grids w/same cosntraints, this is fetching same data multiple times.  Could do something more clever and only pull once.  This seems like a nice spot to multiprocess the sucker too.
                    colnames = []
                    for m in self.metricList[i]:
                        for cn in m.colNameList:
                            colnames.append(cn)
                    if grid.gridtype == 'SPATIAL': 
                        colnames.append(self.config.spatialKey1) 
                        colnames.append(self.config.spatialKey2)
                    colnames = list(set(colnames))
                    self.getData(opsimName,constr, colnames=colnames)
                    #need to add a bit here to calc any needed post-processing columns (e.g., astrometry)
                    gm = self._gridKey(grid)
                    if hasattr(grid,'buildTree'):
                        grid.buildTree(self.data[self.config.spatialKey1],
                                       self.data[self.config.spatialKey2], leafsize=self.config.leafsize)
                    gm.setGrid(grid)
                    gm.runGrid(self.metricList[i], self.data, simDataName=opsimName+'%i'%j, metadata='')
                    gm.reduceAll()
                    gm.plotAll(outDir=self.config.outputDir, savefig=True)
                    gm.writeAll(outDir=self.config.outputDir)
        self.config.save(self.config.outputDir+'/'+'maf_config_asRan.py')
   
                    
if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    driver = MafDriver(configOverrideFilename=configOverrideFilename)
    driver.run()
    
