import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils

def mConfig(config, runName, dbDir='.', outputDir='Out', **kwargs):
    """
    Set up a MAF config for Zeljko's requested metrics. (also included in sstarDriver.py now).
    """
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress = {'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName
    config.outputDir = outputDir

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, WFDpropid, DDpropid = opsimdb.fetchPropIDs()
    
    # Construct a WFD SQL where clause so multiple propIDs can by WFD:
    wfdWhere = ''
    if len(WFDpropid) == 1:
        wfdWhere = "propID = '%s'"%WFDpropid[0]
    else: 
        for i,propid in enumerate(WFDpropid):
            if i == 0:
                wfdWhere = wfdWhere+'('+'propID = %s'%propid
            else:
                wfdWhere = wfdWhere+'or propID = %s'%propid
            wfdWhere = wfdWhere+')'


    # Fetch the total number of visits (to create fraction)
    totalNVisits = opsimdb.fetchNVisits()

    slicerList=[]
    nside = 128

    # fO metrics for all and WFD
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'fO'},
                        plotDict={'units':'Number of Visits', 'xMin':0,
                                    'xMax':1500},
                        summaryStats={'fOArea':{'nside':nside},
                                        'fONv':{'nside':nside}})
    slicer = configureSlicer('fOSlicer', kwargs={'nside':nside},
                            metricDict=makeDict(m1),
                            constraints=['',wfdWhere])
    slicerList.append(slicer)
    
    # Medians in r and i filters
    m1 = configureMetric('MedianMetric', kwargs={'col':'finSeeing'}, summaryStats={'IdentityMetric':{}})
    m2 = configureMetric('MedianMetric', kwargs={'col':'airmass'}, summaryStats={'IdentityMetric':{}})
    m3 = configureMetric('MedianMetric', kwargs={'col':'fiveSigmaDepth'}, summaryStats={'IdentityMetric':{}})
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1,m2,m3),
                            constraints=['filter = "r"', 'filter = "i"'])
    slicerList.append(slicer)


    # Number of visits per proposal
    constraints = ["propID = '%s'"%pid for pid in propids ]
    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Number of Visits Per Proposal'},
                            summaryStats={'IdentityMetric':{}, 'NormalizeMetric':{'normVal':totalNVisits}})
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1),
                            constraints=constraints)
    slicerList.append(slicer)

    # Mean slew time, total number of visits, total visit exposure time, number of nights of observing
    m1 = configureMetric('MeanMetric', kwargs={'col':'slewTime'}, summaryStats={'IdentityMetric':{}})
    m2 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Total Number of Visits'},
                        summaryStats={'IdentityMetric':{}} )
    m3 = configureMetric('OpenShutterMetric', summaryStats={'IdentityMetric':{}} )
    slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1,m2,m3),
                                constraints=[''])
    slicerList.append(slicer)

    m1 = configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'Number of visits per night'}, 
                        summaryStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}})
    m2 = configureMetric('OpenShutterFracMetric',
                         summaryStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}})
    slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'night', 'binsize':1},
                             metricDict=makeDict(m1,m2),
                             constraints=[''])
    slicerList.append(slicer)

    config.slicers = makeDict(*slicerList)
    return config

