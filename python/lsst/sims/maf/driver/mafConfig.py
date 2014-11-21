import lsst.pex.config as pexConfig
import numpy as np

class MixConfig(pexConfig.Config):
    """
    A pexConfig designed to hold a dictionary with string keys and a mix of
    value datatypes.
    """
    plot_str = pexConfig.DictField("", keytype=str, itemtype=str, default={})
    plot_int = pexConfig.DictField("", keytype=str, itemtype=int, default={})
    plot_float = pexConfig.DictField("", keytype=str, itemtype=float, default={})
    plot_bool =  pexConfig.DictField("", keytype=str, itemtype=bool, default={})

class MetricConfig(pexConfig.Config):
    """
    Config object for MAF metrics.
    """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for metrics", dtype=MixConfig, default=None)
    plot = pexConfig.ConfigField("kwargs for plotting parameters", dtype=MixConfig,default=None)
    summaryStats = pexConfig.ConfigDictField("Summary Stats to run", keytype=str,
                                      itemtype=MixConfig,default={})
    histMerge = pexConfig.ConfigField("", dtype=MixConfig, default=None)
    displayDict = pexConfig.ConfigField("How plots should be displayed. keys should be 'displayGroup' w/ value of a string",
                                        dtype=MixConfig,default=None)

class ColStackConfig(pexConfig.Config):
    """
    If there are extra columns that need to be added,
    this config can be used to pass keyword parameters
    """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for stacker", dtype=MixConfig, default=None)
    args = pexConfig.ListField("", dtype=str, default=[])

class MapConfig(pexConfig.Config):
    """
    If there are maps with info that should be passed to each
    """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for maps", dtype=MixConfig, default=None)
    args = pexConfig.ListField("", dtype=str, default=[])


class SlicerConfig(pexConfig.Config):
    """
    Config object for MAF slicers.
    """
    name = pexConfig.Field("", dtype=str, default='')
    kwargs = pexConfig.ConfigField("kwargs for slicer", dtype=MixConfig, default=None)
    metricDict = pexConfig.ConfigDictField(doc="dict of index: metric config", keytype=int,
                                           itemtype=MetricConfig, default={})
    constraints = pexConfig.ListField("", dtype=str, default=[])
    table = pexConfig.Field("Table to use for query", dtype=str, default='Summary')
    stackerDict = pexConfig.ConfigDictField(doc="dict of index: ColstackConfig",
                                          keytype=int, itemtype=ColStackConfig, default={})
    mapsDict = pexConfig.ConfigDictField(doc="dict of index: MapConfig",
                                          keytype=int, itemtype=MapConfig, default={})
    plotConfigs = pexConfig.ConfigDictField(doc="dict of plotConfig objects keyed by metricName", keytype=str,
                                            itemtype=MixConfig, default={})
    metadata = pexConfig.Field("", dtype=str, default='')
    metadataVerbatim = pexConfig.Field("", dtype=bool, default=False)

class MafConfig(pexConfig.Config):
    """
    Using pexConfig to set MAF configuration parameters
    modules: [list] Additional modules to load into MAF (for new metrics, slicers and stackers)
    outputDir: [str] Location where all output files will be written
    figformat:  [str] output figure format (pdf and png are popular)
    dpi:  [int] figure dpi
    opsimName:  [str] string that will be used for output filenames and plot titles
    comment: [str] string added to the output (added to metadata recorded for each metric)
    dbAddress:  [dict] slqAlchamey database address
    verbose:  [boolean] print out timing results
    getConfig:  [boolean] Copy Opsim configuration settings from the database
    slicers:  pexConfig ConfigDictField with slicer configs
    """
    modules = pexConfig.ListField(doc="Optional additional modules to load into MAF", dtype=str, default=[])
    outputDir = pexConfig.Field("Location to write MAF output", str, '')
    figformat = pexConfig.Field("Figure types (png, pdf are popular)", str, 'pdf')
    dpi = pexConfig.Field("Figure dpi", int, 600)
    opsimName = pexConfig.Field("Name to tag output files with", str, 'noName')
    slicers = pexConfig.ConfigDictField(doc="dict of index: slicer config", keytype=int,
                                        itemtype=SlicerConfig, default={})
    comment =  pexConfig.Field("", dtype=str, default='runName')
    dbAddress = pexConfig.DictField("Database access", keytype=str, itemtype=str,
                                    default={'dbAddress':'', 'dbClass':'OpsimDatabase'})
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

def makeDict(*args):
    """Make a dict of index: config from a list of configs
    """
    return dict((ind, config) for ind, config in enumerate(args))

def configureStacker(name, kwargs={}):
    """
    Helper function to generate a Stacker pex config object.
    (configure a stacker)
    """
    config = ColStackConfig()
    config.name = name
    config.kwargs = makeMixConfig(kwargs)
    return config


def configureMap(name, kwargs={}):
    """
    Helper function to generate a map pex config object.
    (configure a stacker)
    """
    config = MapConfig()
    config.name = name
    config.kwargs = makeMixConfig(kwargs)
    return config


def configureMetric(name, kwargs={}, plotDict={}, summaryStats={}, histMerge={}, displayDict={}):
    """
    Helper function to generate a metric pex config object.
    """
    mc = MetricConfig()
    mc.name = name
    mc.summaryStats = {}
    for key in summaryStats:
        mc.summaryStats[key] = makeMixConfig(summaryStats[key])
    mc.histMerge=makeMixConfig(histMerge)
    mc.kwargs = makeMixConfig(kwargs)
    mc.plot = makeMixConfig(plotDict)
    mc.displayDict = makeMixConfig(displayDict)
    return mc

def configureSlicer(name, kwargs={}, metricDict=None, constraints=[''], stackerDict=None,
                    mapsDict=None, metadata='', metadataVerbatim=False, table=None):
    """
    Helper function to generate a Slicer pex config object.
    """
    slicer = SlicerConfig()
    slicer.name = name
    slicer.kwargs = makeMixConfig(kwargs)
    slicer.metadata = metadata
    slicer.metadataVerbatim = metadataVerbatim
    if metricDict:
        slicer.metricDict = metricDict
    slicer.constraints = constraints
    if stackerDict:
        slicer.stackerDict = stackerDict
    if mapsDict:
        slicer.mapsDict = mapsDict
    if table:
        slicer.table = table

    return slicer

def readSlicerConfig(config):
    """Read in a Slicer pex config object """
    name = config.name
    kwargs = readMixConfig(config.kwargs)
    metricDict = config.metricDict
    constraints = config.constraints
    stackerDict = config.stackerDict
    mapsDict = config.mapsDict
    metadata = config.metadata
    metadataVerbatim = config.metadataVerbatim
    return name, kwargs, metricDict, constraints, stackerDict, mapsDict, metadata, metadataVerbatim

def readMetricConfig(config):
    """
    Read in a metric pex config object
    """
    name, kwargs = config2dict(config)
    summaryStats = config.summaryStats
    histMerge = readMixConfig(config.histMerge)
    plotDict = readMixConfig(config.plot)
    displayDict = readMixConfig(config.displayDict)
    return name, kwargs, plotDict, summaryStats, histMerge, displayDict


def readMixConfig(config):
    """
    Read in a pex config object where the different native types have been broken up.
    Returns a dict.
    """
    plotDict={}
    for key in config.plot_str:
        plotDict[key] = config.plot_str[key]
    for key in config.plot_int:
        plotDict[key] = config.plot_int[key]
    for key in config.plot_float:
        plotDict[key] = config.plot_float[key]
    for key in config.plot_bool:
        plotDict[key] = config.plot_bool[key]
    return plotDict


def config2dict(config):
    kwargs = readMixConfig(config.kwargs)
    name = config.name
    return name, kwargs
