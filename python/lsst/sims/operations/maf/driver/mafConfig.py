import lsst.pex.config as pexConfig
   

class MetricConfig(pexConfig.Config):
    name = pexConfig.Field("", dtype=str, default='')   
    kwargs_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    kwargs_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    kwargs_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    kwargs_bool = pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    plot_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    plot_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    plot_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    plot_bool =  pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    params = pexConfig.ListField("", dtype=str, default=[])
    summaryStats=pexConfig.ListField("Summary Stats to run", dtype=str, default=[])

class ColStackConfig(pexConfig.Config):
    """If there are extra columns that need to be added, this config can be used to pass keyword paramters"""
    name = pexConfig.Field("", dtype=str, default='')  
    kwargs_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    kwargs_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    kwargs_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    kwargs_bool = pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    params = pexConfig.ListField("", dtype=str, default=[])


class PlotConfig(pexConfig.Config):
    plot_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    plot_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    plot_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    plot_bool =  pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    
    
class BinnerConfig(pexConfig.Config):
    name = pexConfig.Field("", dtype=str, default='') # Change this to a choiceField? Or do we expect users to make new bins?

    kwargs_str =  pexConfig.DictField("", keytype=str, itemtype=str, default={})
    kwargs_float =  pexConfig.DictField("", keytype=str, itemtype=float, default={})
    kwargs_int =  pexConfig.DictField("", keytype=str, itemtype=int, default={})
    kwargs_bool =  pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    
    params_str =  pexConfig.ListField("", dtype=str, default=[]) 
    params_float =  pexConfig.ListField("", dtype=float, default=[]) 
    params_int =  pexConfig.ListField("", dtype=int, default=[]) 
    params_bool =  pexConfig.ListField("", dtype=bool, default=[])
    
    setupKwargs_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    setupKwargs_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    setupKwargs_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    setupKwargs_bool = pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    
    setupParams_str = pexConfig.ListField("", dtype=str, default=[])
    setupParams_float = pexConfig.ListField("", dtype=float, default=[])
    setupParams_int = pexConfig.ListField("", dtype=int, default=[])
    setupParams_bool = pexConfig.ListField("", dtype=bool, default=[])
  
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig, default={})
    constraints = pexConfig.ListField("", dtype=str, default=[])
    stackCols = pexConfig.ConfigDictField(doc="dict of index: ColstackConfig", keytype=int, itemtype=ColStackConfig, default={}) 
    plotConfigs = pexConfig.ConfigDictField(doc="dict of plotConfig objects keyed by metricName", keytype=str, itemtype=PlotConfig, default={})
    metadata = pexConfig.Field("", dtype=str, default='')
    
class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters"""
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    opsimNames = pexConfig.ListField("Which opsim runs should be analyzed", str, ['opsim_3_61'])
    binners = pexConfig.ConfigDictField(doc="dict of index: binner config", keytype=int, itemtype=BinnerConfig, default={})
    comment =  pexConfig.Field("", dtype=str, default='')
    dbAddress = pexConfig.DictField("Database access", keytype=str, itemtype=str, default={'dbAddress':'','fieldTable':'',  'sessionID':'' , 'proposalTable':'' , 'proposalID':'' })
    
def makeDict(*args):
    """Make a dict of index: config from a list of configs
    """
    return dict((ind, config) for ind, config in enumerate(args))


def makeBinnerConfig(name, params=[], kwargs={}, setupParams=[], setupKwargs={}, metricDict=None, constraints=[], stackCols=None, plotConfigs=None, metadata=''):
    binner = BinnerConfig()
    binner.name = name
    binner.metadata=metadata
    if metricDict:  binner.metricDict=metricDict
    binner.constraints=constraints
    if stackCols: binner.stackCols = stackCols
    if plotConfigs:  binner.plotConfigs = plotConfigs
    binner.params_str=[]
    binner.params_float=[]
    binner.params_int = []
    binner.params_bool=[]
    for p in params:
        if type(p) is str:
            binner.params_str.append(p)
        elif type(p) is float:
            binner.params_float.append(p)
        elif type(p) is int:
            binner.params_int.append(p)
        elif type(p) is bool:
            binner.params_bool.append(p)
        else:
            raise Exception('Unsupported parameter data type')

    for key in kwargs.keys():
        if type(kwargs[key]) is str:
            binner.kwargs_str[key] = kwargs[key]
        elif type(kwargs[key]) is float:
            binner.kwargs_float[key] = kwargs[key]
        elif type(kwargs[key]) is int:
            binner.kwargs_float[key] = kwargs[key]
        elif type(kwargs[key]) is bool:
            binner.kwargs_bool[key] = kwargs[key]
        else:
            raise Exception('Unsupported kwarg data type')

    binner.setupParams_str=[]
    binner.setupParams_float=[]
    binner.setupParams_int = []
    binner.setupParams_bool=[]

    for p in setupParams:
        if type(p) is str:
            binner.setupParams_str.append(p)
        elif type(p) is float:
            binner.setupParams_float.append(p) 
        elif type(p) is int:
            binner.setupParams_int.append(p) 
        elif type(p) is bool:
            binner.setupParams_bool.append(p) 
        else:
            raise Exception('Unsupported parameter data type')

    for key in setupKwargs.keys():
        if type(setupKwargs[key]) is str:
            binner.setupKwargs_str[key] = setupKwargs[key]
        elif type(setupKwargs[key]) is float:
            binner.setupKwargs_float[key] = setupKwargs[key]
        elif type(setupKwargs[key]) is int:
            binner.setupKwargs_float[key] = setupKwargs[key]
        elif type(setupKwargs[key]) is bool:
            binner.setupKwargs_bool[key] = setupKwargs[key]
        else:
            raise Exception('Unsupported kwarg data type')

    return binner

def readBinnerConfig(config):
    name = config.name
    params = []
    params.append(config.params_str)
    params.append(config.params_float)
    params.append(config.params_int)
    params.append(config.params_bool)
    params = filter(None,params)
    kwargs = {}
    for key in config.kwargs_str:  kwargs[key] = config.kwargs_str[key]
    for key in config.kwargs_float:  kwargs[key] = config.kwargs_float[key]
    for key in config.kwargs_int:  kwargs[key] = config.kwargs_int[key]
    for key in config.kwargs_bool:  kwargs[key] = config.kwargs_bool[key]
    setupParams=[]
    setupParams.append(config.setupParams_str)
    setupParams.append(config.setupParams_float)
    setupParams.append(config.setupParams_int)
    setupParams.append(config.setupParams_bool)
    setupParams = filter(None,setupParams)
    setupKwargs = {}
    for key in config.setupKwargs_str:  setupKwargs[key] = config.setupKwargs_str[key]
    for key in config.setupKwargs_float:  setupKwargs[key] = config.setupKwargs_float[key]
    for key in config.setupKwargs_int:  setupKwargs[key] = config.setupKwargs_int[key]
    for key in config.setupKwargs_bool:  setupKwargs[key] = config.setupKwargs_bool[key]

    metricDict = config.metricDict    
    constraints=config.constraints
    stackCols= config.stackCols
    plotConfigs = config.plotConfigs
    metadata=config.metadata
    return name, params, kwargs, setupParams,setupKwargs, metricDict, constraints, stackCols, plotConfigs, metadata
        
def makeMetricConfig(name, params=[], kwargs={}, plotDict={}, summaryStats=[]):
    mc = MetricConfig()
    mc.name = name
    mc.params=params
    mc.summaryStats = summaryStats
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
    for key in plotDict.keys():
        if type(plotDict[key]) is str:
            mc.plot_str[key] = plotDict[key]
        elif type(plotDict[key]) is float:
            mc.plot_float[key] = plotDict[key]
        elif type(plotDict[key]) is int:
            mc.plot_int[key] = plotDict[key]
        elif type(plotDict[key]) is bool:
            mc.plot_bool[key] = plotDict[key]
        else:
            raise Exception('Unsupported kwarg data type')
    return mc


def makePlotConfig(plotDict):
    mc = PlotConfig()
    for key in plotDict.keys():
        if type(plotDict[key]) is str:
            mc.plot_str[key] = plotDict[key]
        elif type(plotDict[key]) is float:
            mc.plot_float[key] = plotDict[key]
        elif type(plotDict[key]) is int:
            mc.plot_int[key] = plotDict[key]
        elif type(plotDict[key]) is bool:
            mc.plot_bool[key] = plotDict[key]
        else:
            raise Exception('Unsupported kwarg data type')
    return mc

def readPlotConfig(config):
    plotDict={}
    for key in config.plot_str:  plotDict[key] = config.plot_str[key]
    for key in config.plot_int:  plotDict[key] = config.plot_int[key]
    for key in config.plot_float:  plotDict[key] = config.plot_float[key]
    for key in config.plot_bool:  plotDict[key] = config.plot_bool[key]
    return plotDict
 
    
def readMetricConfig(config):
    name, params,kwargs = config2dict(config)
    plotDict={}
    summaryStats = config.summaryStats
    for key in config.plot_str:  plotDict[key] = config.plot_str[key]
    for key in config.plot_int:  plotDict[key] = config.plot_int[key]
    for key in config.plot_float:  plotDict[key] = config.plot_float[key]
    for key in config.plot_bool:  plotDict[key] = config.plot_bool[key]
    return name,params,kwargs,plotDict,summaryStats
   


def config2dict(config):
    kwargs={}
    for key in config.kwargs_str:  kwargs[key] = config.kwargs_str[key]
    for key in config.kwargs_int:  kwargs[key] = config.kwargs_int[key]
    for key in config.kwargs_float:  kwargs[key] = config.kwargs_float[key]
    for key in config.kwargs_bool:  kwargs[key] = config.kwargs_bool[key]
    params=config.params
    name = config.name
    return name, params, kwargs
 
     
                                        

                                      
