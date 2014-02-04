import numpy as np
import lsst.pex.config as pexConfig

class MetricConfig(pexConfig.Config):
    metric = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.DictField("", keytype=str, itemtype=float, default={}) 
    params = pexConfig.ListField("", dtype=str, default=[]) 

class BinnerConfig(pexConfig.Config):
    binner = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new bins?
    kwargs =  pexConfig.DictField("", keytype=str, itemtype=float, default={}) 
    params =  pexConfig.ListField("", dtype=str, default=[]) 
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig, default={})
    spatialKey1 =  pexConfig.Field("", dtype=str, default='')
    spatialKey2 =  pexConfig.Field("", dtype=str, default='')
    leafsize = pexConfig.Field("Leaf size for kdtree", float, 100)
    constraints = pexConfig.ListField("", dtype=str, default=[''])
   
class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters"""
    dbAddress = pexConfig.Field("Address to the database to query." , str, '')
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    opsimNames = pexConfig.ListField("Which opsim runs should be analyzed", str, ['opsim_3_61'])
    binners = pexConfig.ConfigDictField(doc="dict of index: binner config", keytype=int, itemtype=BinnerConfig, default={}) 

def makeDict(*args):
    """Make a dict of index: config from a list of configs
    """
    return dict((ind, config) for ind, config in enumerate(args))

def makeMetricConfig(name, params=[], kwargs={}):
    mc = MetricConfig()
    mc.metric = name
    mc.kwargs=kwargs
    mc.params=params
    return mc

 
                                        

                                      
