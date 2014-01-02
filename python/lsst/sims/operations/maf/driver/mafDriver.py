import numpy as np #segfault if numpy not imported 1st, argle bargle!
from mafConfig import MafConfig
import lsst.sims.operations.maf.grids as grids
import lsst.sims.operations.maf.metrics as metrics

def MafDriver(configOverrideFilename=None):
    """Script for configuring and running metrics on Opsim output """
    config=MafConfig()

    # Load any config file
    if configOverrideFilename is not None:
        config.load(configOverrideFilename)

    # Load any parameters set on the command line
    
    # Validate and freeze the config
    config.validate()
    config.freeze()

    # Construct the grid and metric objects
    s_grids = [config.grid1,config.grid2,config.grid3,config.grid4,config.grid5,
               config.grid6,config.grid7,config.grid8,config.grid9,config.grid10]
    s_gridKwrds = [config.kwrdsForGrid1,config.kwrdsForGrid2,config.
                   kwrdsForGrid3,config.kwrdsForGrid4,config.kwrdsForGrid5,
                   config.kwrdsForGrid6,config.kwrdsForGrid7,config.kwrdsForGrid8,
                   config.kwrdsForGrid9,config.kwrdsForGrid10 ]
    s_metrics = [config.metricsForGrid1,config.metricsForGrid2, config.metricsForGrid3,
                 config.metricsForGrid4, config.metricsForGrid5, config.metricsForGrid6,
                 config.metricsForGrid7, config.metricsForGrid8, config.metricsForGrid9,
                 config.metricsForGrid10]
    s_metricParams = [config.metricParamsForGrid1,config.metricParamsForGrid2
                      ,config.metricParamsForGrid3,config.metricParamsForGrid4,config.metricParamsForGrid5,
                      config.metricParamsForGrid6,config.metricParamsForGrid7,config.metricParamsForGrid8,
                      config.metricParamsForGrid9,config.metricParamsForGrid10]
    s_metricKwrds = [config.metricKwrdsForGrid1,config.metricKwrdsForGrid2,config.metricKwrdsForGrid3,
                     config.metricKwrdsForGrid4,config.metricKwrdsForGrid5,
                     config.metricKwrdsForGrid6,config.metricKwrdsForGrid7,
                     config.metricKwrdsForGrid8,config.metricKwrdsForGrid9,config.metricKwrdsForGrid10]
    constraints = [config.constraintsForGrid1,config.constraintsForGrid2,config.constraintsForGrid3,
                   config.constraintsForGrid4,config.constraintsForGrid5,config.constraintsForGrid6,
                   config.constraintsForGrid7,config.constraintsForGrid8,config.constraintsForGrid9,
                   config.constraintsForGrid10]

    gridList=[]
    metricList=[]
    
    for i,s_grid in enumerate(s_grids):
        if s_grid is not '':
            gridList.append(getattr(grids,s_grid)(**eval('dict('+ s_gridKwrds[i]+')'))   )
            sub_metricList=[]
            for j,s_metric in enumerate(s_metrics[i]):
                sub_metricList.append(getattr(metrics,s_metric)(*s_metricParams[i][j].split(','), **eval('dict('+s_metricKwrds[i][j]+')' )))
            metricList.append(sub_metricList)

    
    # Loop over each grid in gridList:  for i,grid in enumerate(gridList)
        # Loop over Opsim run names
            # Loop over each SQL-constraint for that grid:  for j,constr in enumerate(constraints[i])
                # Pull data from db using constraints[i][j] and Opsim run name
                # build kd-tree if needed--maybe make ra,dec keys and leafsize keywords for HealpixGrid so easier to pass in.
                # make gridmetric object
                # gridmetric.setGrid(gridList[i])
                # gridmetric.runGrid(metricList[i], data, yadda yadda)
                # gridmetric.reduceAll() ; gridmetric.plotAll(), gridmetric.writeAll()

if __name__ == "__main__":
    import sys
    configOverrideFilename = sys.argv[1]
    MafDriver(configOverrideFilename)
    
