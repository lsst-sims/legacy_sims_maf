import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils
import lsst.sims.maf.stackers as stackers

def mConfig(config, runName, dbDir='.', outputDir='Dithers', nside=128, **kwargs):
    """
    A MAF config for analysis of various dithers applied to an opsim run.

    runName must correspond to the name of the opsim output
        (minus '_sqlite.db', although if added this will be stripped off)

    dbDir is the directory containing the sqlite database
    outputDir is the output directory
    """
    # To use the maf contributed metrics (including galaxy counts metric).
    config.modules = ['mafContrib']

    # Setup Database access
    config.outputDir = outputDir
    if runName.endswith('_sqlite.db'):
        runName = runName.replace('_sqlite.db', '')
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName
    config.figformat = 'pdf'

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, propTags = opsimdb.fetchPropInfo()
    if 'WFD' not in propTags:
        propTags['WFD'] = []

    # Construct a WFD SQL where clause (handles the case of multiple propIDs mapping into "WFD").
    wfdWhere = utils.createSQLWhere('WFD', propTags)
    print 'WFD "where" clause: %s' %(wfdWhere)

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    summaryStats = {'MeanMetric':{}, 'MedianMetric':{}, 'RmsMetric':{},
                    'PercentileMetric 1':{'metricName':'25th%ile', 'percentile':25},
                    'PercentileMetric 2':{'metricName':'75th%ile', 'percentile':75},
                    'MinMetric':{},
                    'MaxMetric':{},
                    'TotalPowerMetric':{}}

    ditherDict = {}
    ditherDict['NoDither'] = None
    ditherDict['HexDither'] = None
    ditherDict['RandomDither'] = stackers.RandomDitherStacker()
    ditherDict['NightlyRandomDither'] = stackers.NightlyRandomDitherStacker()
    ditherDict['SpiralDither'] = stackers.SpiralDitherStacker()
    ditherDict['NightlySpiralDither'] = stackers.NightlySpiralDitherStacker()
    ditherDict['SequentialHexDither'] = stackers.SequentialHexDitherStacker()
    ditherDict['NightlySequentialHexDither'] = stackers.NightlySequentialHexDitherStacker()
    dithernames = ['NoDither', 'HexDither', 'SequentialHexDither', 'NightlySequentialHexDither',
                   'RandomDither', 'NightlyRandomDither', 'SpiralDither', 'NightlySpiralDither']

    filterlist = ['u', 'g', 'r', 'i', 'z', 'y']

    # Set up sqlconstraint and use this as the outermost loop.
    slicerList = []
    order = 0
    for f in filterlist:
        sqlconstraint = "filter = '%s'" %(f)
        sqlconstraint += ' and %s' %(wfdWhere)
        common_metadata = '%s band WFD' %(f)
        histNum = 0
        # Set up metrics to run for each dither pattern.
        # Set up appropriate slicer and update order/hist label for each ditherpattern, using the same sqlconstraint.
        for i, dithername in enumerate(dithernames):
            if dithername == 'NoDither':
                racol = 'fieldRA'
                deccol = 'fieldDec'
            elif dithername == 'HexDither':
                racol = 'ditheredRA'
                deccol = 'ditheredDec'
            else:
                racol = ditherDict[dithername].colsAdded[0]
                deccol = ditherDict[dithername].colsAdded[1]
            metadata = common_metadata + ' ' + dithername

            metricList = []
            metricList.append(configureMetric('Coaddm5Metric',
                                            plotDict={'units':'Coadded m5'},
                                            summaryStats=summaryStats,
                                            displayDict={'group':'Dithers', 'subgroup':'Coadded Depth',
                                                         'order':i+filtorder[f]*100},
                                            histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s' %(metadata),
                                                         'legendloc':'upper right'}))
            histNum += 1
            metricList.append(configureMetric('CountUniqueMetric', kwargs={'col':'night', 'metricName':'Unique Nights'},
                                               plotDict={'title':'Number of unique nights with observations',
                                                         'cbarFormat':'%d'},
                                                summaryStats=summaryStats,
                                                displayDict={'group':'Dithers', 'subgroup':'Number of nights',
                                                             'order':i+filtorder[f]*100},
                                                histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s' %(metadata),
                                                         'legendloc':'upper right'}))
            histNum += 1
            metricList.append(configureMetric('CountMetric', kwargs={'col':'expMJD',
                                                                 'metricName':'Number of Visits'},
                                          plotDict={'title':'Number of visits',
                                                    'colorMin':0, 'colorMax':300,
                                                    'cbarFormat': '%d'},
                                            summaryStats=summaryStats,
                                            displayDict={'group':'Dithers','subgroup':'Number of visits',
                                                         'order':i+filtorder[f]*100},
                                            histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s' %(metadata),
                                                         'legendloc':'upper right'}))
            histNum += 1
            metricList.append(configureMetric('mafContrib.GalaxyCountsMetric', kwargs={'nside':nside},
                                          summaryStats=summaryStats,
                                            displayDict={'group':'Dithers', 'subgroup':'Galaxy Counts',
                                                              'order':i+filtorder[f]*100},
                                            histMerge={'histNum':histNum, 'color':colors[f], 'label':'%s' %(metadata),
                                                         'legendloc':'upper right'}))
            histNum += 1
            slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside, 'spatialkey1':racol, 'spatialkey2':deccol},
                                    constraints=[sqlconstraint], metricDict=makeDict(*metricList),
                                    metadata= metadata, metadataVerbatim=True)
            slicerList.append(slicer)

    config.slicers = makeDict(*slicerList)
    return config
