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
    args = pexConfig.ListField("", dtype=str, default=[])
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
    args = pexConfig.ListField("", dtype=str, default=[])
    
class SlicerConfig(pexConfig.Config):
    """Config object for MAF slicers """
    name = pexConfig.Field("", dtype=str, default='') 

    kwargs = pexConfig.ConfigField("kwargs for slicer", dtype=MixConfig, default=None)

    args_str =  pexConfig.ListField("", dtype=str, default=[]) 
    args_float =  pexConfig.ListField("", dtype=float, default=[]) 
    args_int =  pexConfig.ListField("", dtype=int, default=[]) 
    args_bool =  pexConfig.ListField("", dtype=bool, default=[])

    setupKwargs = pexConfig.ConfigField("setup kwargs for slicer", dtype=MixConfig, default=None)
    
    setupArgs_str = pexConfig.ListField("", dtype=str, default=[])
    setupArgs_float = pexConfig.ListField("", dtype=float, default=[])
    setupArgs_int = pexConfig.ListField("", dtype=int, default=[])
    setupArgs_bool = pexConfig.ListField("", dtype=bool, default=[])
  
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int, itemtype=MetricConfig,
                                           default={})
    constraints = pexConfig.ListField("", dtype=str, default=[])
    stackCols = pexConfig.ConfigDictField(doc="dict of index: ColstackConfig", keytype=int, itemtype=ColStackConfig,
                                          default={}) 
    plotConfigs = pexConfig.ConfigDictField(doc="dict of plotConfig objects keyed by metricName", keytype=str,
                                            itemtype=MixConfig, default={})
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
    slicers = pexConfig.ConfigDictField(doc="dict of index: slicer config", keytype=int,
                                        itemtype=SlicerConfig, default={})
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

def configureSlicer(name, args=[], kwargs={}, setupArgs=[], setupKwargs={},
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
    slicer.args_str=[]
    slicer.args_float=[]
    slicer.args_int = []
    slicer.args_bool=[]

    for p in args:
        if type(p) is str:
            slicer.args_str.append(p)
        elif type(p) is float:
            slicer.args_float.append(p)
        elif type(p) is int:
            slicer.args_int.append(p)
        elif type(p) is bool:
            slicer.args_bool.append(p)
        else:
            raise Exception('Unsupported parameter data type')

    slicer.kwargs = makeMixConfig(kwargs)

    slicer.setupArgs_str=[]
    slicer.setupArgs_float=[]
    slicer.setupArgs_int = []
    slicer.setupArgs_bool=[]

    for p in setupArgs:
        if type(p) is str:
            slicer.setupArgs_str.append(p)
        elif type(p) is float:
            slicer.setupArgs_float.append(p) 
        elif type(p) is int:
            slicer.setupArgs_int.append(p) 
        elif type(p) is bool:
            slicer.setupArgs_bool.append(p) 
        else:
            raise Exception('Unsupported parameter data type')
    slicer.setupKwargs = makeMixConfig(setupKwargs)

    return slicer

def readSlicerConfig(config):
    """Read in a Slicer pex config object """
    name = config.name
    args = []
    args.append(config.args_str)
    args.append(config.args_float)
    args.append(config.args_int)
    args.append(config.args_bool)
    args = filter(None,args)
    kwargs = readMixConfig(config.kwargs)
    setupArgs=[]
    setupArgs.append(config.setupArgs_str)
    setupArgs.append(config.setupArgs_float)
    setupArgs.append(config.setupArgs_int)
    setupArgs.append(config.setupArgs_bool)
    setupArgs = filter(None,setupArgs)
    setupKwargs = readMixConfig(config.setupKwargs)
    metricDict = config.metricDict    
    constraints=config.constraints
    stackCols= config.stackCols
    plotConfigs = config.plotConfigs
    metadata=config.metadata
    return name, args, kwargs, setupArgs,setupKwargs, metricDict, constraints, stackCols, plotConfigs, metadata
        
def configureMetric(name, args=[], kwargs={}, plotDict={}, summaryStats={}, histMerge={}):
    """
    Helper function to generate a metric pex config object.
    """
    mc = MetricConfig()
    mc.name = name
    mc.args=args
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
    name, args,kwargs = config2dict(config)
    summaryStats = config.summaryStats
    histMerge = readMixConfig(config.histMerge)
    plotDict = readMixConfig(config.plot)
    return name,args,kwargs,plotDict,summaryStats, histMerge

def config2dict(config):
    kwargs = readMixConfig(config.kwargs)
    args=config.args
    name = config.name
    return name, args, kwargs
 
     
                                        

                                      
