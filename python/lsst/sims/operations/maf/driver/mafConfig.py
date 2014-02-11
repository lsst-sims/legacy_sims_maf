import numpy as np
import lsst.pex.config as pexConfig
   

class MetricConfig(pexConfig.Config):
    name = pexConfig.Field("", dtype=str, default='')   
    kwargs_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    kwargs_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    kwargs_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    kwargs_bool = pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    params = pexConfig.ListField("", dtype=str, default=[])

class ColStackConfig(pexConfig.Config):
    """If there are extra columns that need to be added, this config can be used to pass keyword paramters"""
    name = pexConfig.Field("", dtype=str, default='')  
    kwargs_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    kwargs_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    kwargs_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    kwargs_bool = pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    params = pexConfig.ListField("", dtype=str, default=[])


class BinnerConfig(pexConfig.Config):
    name = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new bins?
    kwargs =  pexConfig.DictField("", keytype=str, itemtype=float, default={}) 
    params =  pexConfig.ListField("", dtype=str, default=[]) 
    setupKwargs = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    setupParams = pexConfig.ListField("", dtype=str, default=[])
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig, default={})
    constraints = pexConfig.ListField("", dtype=str, default=[''])
    stackCols = pexConfig.ConfigDictField(doc="dict of index: ColstackConfig", keytype=int, itemtype=ColStackConfig, default={}) 

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
    mc.name = name
    mc.params=params
    # Break the kwargs by data type
    for key in kwargs.keys():
        if type(kwargs[key]) is str:
            mc.kwargs_str[key] = kwargs[key]
        elif type(kwargs[key]) is float:
            mc.kwargs_float[key] = kwargs[key]
        elif type(kwargs[key]) is int:
            mc.kwargs_float[key] = kwargs[key]
        elif type(kwargs[key]) is bool:
            mc.kwargs_bool[key] = kwargs[key]
        else:
            raise Exception('Unsupported kwarg data type')
    return mc

def config2dict(config):
    kwargs={}
    for key in config.kwargs_str:  kwargs[key] = config.kwargs_str[key]
    for key in config.kwargs_int:  kwargs[key] = config.kwargs_int[key]
    for key in config.kwargs_float:  kwargs[key] = config.kwargs_float[key]
    for key in config.kwargs_bool:  kwargs[key] = config.kwargs_bool[key]
    params=config.params
    name = config.name
    return name, params, kwargs
 
     
                                        

                                      
