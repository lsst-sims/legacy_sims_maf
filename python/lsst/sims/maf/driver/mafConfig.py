import lsst.pex.config as pexConfig
import numpy as np

class MixConfig(pexConfig.Config):
    """A pexConfig designed to hold a dictionary with string keys and a mix of 
    value datatypes."""
    plot_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    plot_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    plot_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    plot_bool =  pexConfig.DictField("", keytype=str, itemtype=bool, default={})
    
class MetricConfig(pexConfig.Config):
    """Config object for MAF metrics """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for metrics", dtype=MixConfig, default=None)
    plot = pexConfig.ConfigField("kwargs for plotting parameters", dtype=MixConfig,default=None)
    params = pexConfig.ListField("", dtype=str, default=[])
    summaryStats=pexConfig.ConfigDictField("Summary Stats to run", keytype=str, 
                                      itemtype=MixConfig,default={})
    histMerge = pexConfig.ConfigField("", dtype=MixConfig, default=None)

class ColStackConfig(pexConfig.Config):
    """
    If there are extra columns that need to be added,
    this config can be used to pass keyword parameters
    """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for stacker", dtype=MixConfig, default=None)
    params = pexConfig.ListField("", dtype=str, default=[])
    
class SlicerConfig(pexConfig.Config):
    """Config object for MAF slicers """
    name = pexConfig.Field("", dtype=str, default='') 

    kwargs = pexConfig.ConfigField("kwargs for slicer", dtype=MixConfig, default=None)

    params_str =  pexConfig.ListField("", dtype=str, default=[]) 
    params_float =  pexConfig.ListField("", dtype=float, default=[]) 
    params_int =  pexConfig.ListField("", dtype=int, default=[]) 
    params_bool =  pexConfig.ListField("", dtype=bool, default=[])

    setupKwargs = pexConfig.ConfigField("setup kwargs for slicer", dtype=MixConfig, default=None)
    
    setupParams_str = pexConfig.ListField("", dtype=str, default=[])
    setupParams_float = pexConfig.ListField("", dtype=float, default=[])
    setupParams_int = pexConfig.ListField("", dtype=int, default=[])
    setupParams_bool = pexConfig.ListField("", dtype=bool, default=[])
  
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig, default={})
    constraints = pexConfig.ListField("", dtype=str, default=[])
    stackCols = pexConfig.ConfigDictField(doc="dict of index: ColstackConfig", keytype=int, itemtype=ColStackConfig, default={}) 
    plotConfigs = pexConfig.ConfigDictField(doc="dict of plotConfig objects keyed by metricName", keytype=str, itemtype=MixConfig, default={})
    metadata = pexConfig.Field("", dtype=str, default='')

class MafConfig(pexConfig.Config):
    """Using pexConfig to set MAF configuration parameters
    modules: Additional modules to load into MAF (for new metrics, slicers and stackers)
    outputDir:  Location where all output files will be written
    figformat:  output figure format (pdf and png are popular)
    dpi:  figure dpi
    opsimName:  string that will be used for output filenames and plot titles
    slicers:  pexConfig ConfigDictField with slicer configs
    comment:  string added to the output
    dbAddress:  slqAlchamey database address
    verbose:  print out timing results
    getConfig:  Copy Opsim configuration settings from the database
    """
    modules = pexConfig.ListField(doc="Optional additional modules to load into MAF", dtype=str, default=[])
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    figformat = pexConfig.Field("Figure types (png, pdf are popular)", str, 'pdf')
    dpi = pexConfig.Field("Figure dpi", int, 600)
    opsimName = pexConfig.Field("Name to tag output files with", str, 'noName')
    slicers = pexConfig.ConfigDictField(doc="dict of index: slicer config", keytype=int, itemtype=SlicerConfig, default={})
    comment =  pexConfig.Field("", dtype=str, default='runName')
    dbAddress = pexConfig.DictField("Database access", keytype=str, itemtype=str, default={'dbAddress':''})
    verbose = pexConfig.Field("", dtype=bool, default=False)
    getConfig = pexConfig.Field("", dtype=bool, default=True)

    
def makeMixConfig(plotDict):
    """Helper function to convert a dictionary into a MixConfig.  
    Input dictionary must have str keys and values that are str, float, int, or bool.
    If the input dict has numpy data types, they are converted to similar native python types."""
    mc = MixConfig()
    for key in plotDict.keys():
        if type(plotDict[key]).__module__ == np.__name__:
            value = np.asscalar(plotDict[key])
        else:
            value = plotDict[key]
        if type(plotDict[key]) is str:
            mc.plot_str[key] = value
        elif type(plotDict[key]) is float:
            mc.plot_float[key] = value
        elif type(plotDict[key]) is int:
            mc.plot_int[key] = value
        elif type(plotDict[key]) is bool:
            mc.plot_bool[key] = value
        else:
            print '%s has data type %s'%(key,type(value).__name__)
            raise Exception('Unsupported kwarg data type')
    return mc

def configureStacker(name, kwargs={}):
    """Configure a column stacker."""
    config = ColStackConfig()
    config.name = name
    config.kwargs = makeMixConfig(kwargs)
    return config
    
def makeDict(*args):
    """Make a dict of index: config from a list of configs
    """
    return dict((ind, config) for ind, config in enumerate(args))

def configureSlicer(name, params=[], kwargs={}, setupParams=[], setupKwargs={},
                    metricDict=None, constraints=[], stackCols=None, plotConfigs=None, metadata=''):
    """
    Helper function to generate a Slicer pex config object
    """
    slicer = SlicerConfig()
    slicer.name = name
    slicer.metadata=metadata
    if metricDict:  slicer.metricDict=metricDict
    slicer.constraints=constraints
    if stackCols: slicer.stackCols = stackCols
    if plotConfigs:  slicer.plotConfigs = plotConfigs
    slicer.params_str=[]
    slicer.params_float=[]
    slicer.params_int = []
    slicer.params_bool=[]

    for p in params:
        if type(p) is str:
            slicer.params_str.append(p)
        elif type(p) is float:
            slicer.params_float.append(p)
        elif type(p) is int:
            slicer.params_int.append(p)
        elif type(p) is bool:
            slicer.params_bool.append(p)
        else:
            raise Exception('Unsupported parameter data type')

    slicer.kwargs = makeMixConfig(kwargs)

    slicer.setupParams_str=[]
    slicer.setupParams_float=[]
    slicer.setupParams_int = []
    slicer.setupParams_bool=[]

    for p in setupParams:
        if type(p) is str:
            slicer.setupParams_str.append(p)
        elif type(p) is float:
            slicer.setupParams_float.append(p) 
        elif type(p) is int:
            slicer.setupParams_int.append(p) 
        elif type(p) is bool:
            slicer.setupParams_bool.append(p) 
        else:
            raise Exception('Unsupported parameter data type')
    slicer.setupKwargs = makeMixConfig(setupKwargs)

    return slicer

def readSlicerConfig(config):
    """Read in a Slicer pex config object """
    name = config.name
    params = []
    params.append(config.params_str)
    params.append(config.params_float)
    params.append(config.params_int)
    params.append(config.params_bool)
    params = filter(None,params)
    kwargs = readMixConfig(config.kwargs)
    setupParams=[]
    setupParams.append(config.setupParams_str)
    setupParams.append(config.setupParams_float)
    setupParams.append(config.setupParams_int)
    setupParams.append(config.setupParams_bool)
    setupParams = filter(None,setupParams)
    setupKwargs = readMixConfig(config.setupKwargs)
    metricDict = config.metricDict    
    constraints=config.constraints
    stackCols= config.stackCols
    plotConfigs = config.plotConfigs
    metadata=config.metadata
    return name, params, kwargs, setupParams,setupKwargs, metricDict, constraints, stackCols, plotConfigs, metadata
        
def configureMetric(name, params=[], kwargs={}, plotDict={}, summaryStats={}, histMerge={}):
    """
    Helper function to generate a metric pex config object.
    """
    mc = MetricConfig()
    mc.name = name
    mc.params=params
    mc.summaryStats = {}
    for key in summaryStats:
        mc.summaryStats[key] = makeMixConfig(summaryStats[key])
    mc.histMerge=makeMixConfig(histMerge)
    mc.kwargs = makeMixConfig(kwargs)
    mc.plot = makeMixConfig(plotDict)
    return mc

def readMixConfig(config):
    """
    Read in a pex config object where the different native types have been broken up.
    Returns a dict.
    """
    plotDict={}
    for key in config.plot_str:  plotDict[key] = config.plot_str[key]
    for key in config.plot_int:  plotDict[key] = config.plot_int[key]
    for key in config.plot_float:  plotDict[key] = config.plot_float[key]
    for key in config.plot_bool:  plotDict[key] = config.plot_bool[key]
    return plotDict
    
def readMetricConfig(config):
    """
    Read in a metric pex config object
    """
    name, params,kwargs = config2dict(config)
    summaryStats = config.summaryStats
    histMerge = readMixConfig(config.histMerge)
    plotDict = readMixConfig(config.plot)
    return name,params,kwargs,plotDict,summaryStats, histMerge

def config2dict(config):
    kwargs = readMixConfig(config.kwargs)
    params=config.params
    name = config.name
    return name, params, kwargs
 
     
                                        

                                      
