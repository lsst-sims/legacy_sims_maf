import numpy as np
import lsst.pex.config as pexConfig

class GridPackConfig(pexConfig.Config):
    """Config for packaging up a grid and list of metrics to run together """
    grid = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new grids? 
    kwrds = pexConfig.Field("", dtype=str, default='')
    metrics = pexConfig.ListField("", dtype=str, default=[''])
    constraints = pexConfig.ListField("", dtype=str, default=[''])
    metricParams = pexConfig.ListField("", dtype=str, default=[''])
    metricKwrds = pexConfig.ListField("", dtype=str, default=[''])

class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters"""
    dbAddress = pexConfig.Field("Address to the database to query." , str, '')
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    opsimNames = pexConfig.ListField("Which opsim runs should be analyzed", str, ['opsim_3_61'])

    spatialKey1 = pexConfig.Field("first key to build KD tree on", str, 'fieldRA')
    spatialKey2 =pexConfig.Field("first key to build KD tree on", str, 'fieldDec')
    leafsize = pexConfig.Field("leafsize for KD tree", float, 5000)
    # Manually unrolling loop since pex can't take a list of lists.
    from lsst.sims.operations.maf.driver.mafConfig import GridPackConfig
    grid1 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid2 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid3 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid4 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid5 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid6 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid7 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid8 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid9 = pexConfig.ConfigField("",GridPackConfig, default=None)
    grid10 = pexConfig.ConfigField("",GridPackConfig, default=None)


    
