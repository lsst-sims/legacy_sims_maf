import numpy as np
import lsst.pex.config as pexConfig


class MetricConfig(pexConfig.Config):
    metric = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.Field("", dtype=str, default='')
    params = pexConfig.Field("", dtype=str, default='')

class BinnerConfig(pexConfig.Config):
    binner = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new bins?
    kwargs =  pexConfig.Field("", dtype=str, default='')
    params =  pexConfig.Field("", dtype=str, default='')
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig, default={})

class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters"""
    dbAddress = pexConfig.Field("Address to the database to query." , str, '')
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    opsimNames = pexConfig.ListField("Which opsim runs should be analyzed", str, ['opsim_3_61'])
    constraints = pexConfig.ListField("", dtype=str, default=[''])
    binners = pexConfig.ConfigDictField(doc="dict of index: binner config", keytype=int, itemtype=BinnerConfig, default={}) 

def makeDict(*args):
    """Make a dict of index: config from a list of configs
    """
    return dict((ind, config) for ind, config in enumerate(args))

def makeMetricConfig(name, params='', kwargs=''):
    mc = MetricConfig()
    mc.metric = name
    mc.kwargs=kwargs
    mc.params=params
    return mc

 
                                        

                                      
class BinPackConfig(pexConfig.Config):
    """Config for packaging up a grid and list of metrics to run together """
    binner = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new bins? 
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
    from lsst.sims.operations.maf.driver.mafConfig import BinPackConfig
    bin1 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin2 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin3 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin4 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin5 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin6 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin7 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin8 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin9 = pexConfig.ConfigField("",BinPackConfig, default=None)
    bin10 = pexConfig.ConfigField("",BinPackConfig, default=None)


    
