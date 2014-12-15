# A MAF config that replicates the SSTAR plots

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mConfig(config, runName, dbDir='.', outputDir='Out', slicerName='HealpixSlicer',
            benchmark='design', **kwargs):
    """
    A MAF config for SSTAR-like analysis of an opsim run.

    runName must correspond to the name of the opsim output
        (minus '_sqlite.db', although if added this will be stripped off)

    dbDir is the directory the database resides in

    outputDir is the output directory for MAF

    Uses 'slicerName' for metrics which have the option of using
      [HealpixSlicer, OpsimFieldSlicer, or HealpixSlicerDither]
      (dithered healpix slicer uses ditheredRA/dec values).

    Uses 'benchmark' (which can be design or stretch) to scale plots of number of visits and coadded depth.
    """

    # Setup Database access
    config.outputDir = outputDir
    if runName.endswith('_sqlite.db'):
        runName = runName.replace('_sqlite.db', '')
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName
    config.figformat = 'pdf'

    config.verbose = True

    opsimdb = utils.connectOpsimDb(config.dbAddress)
    totalNVisits = opsimdb.fetchNVisits()
    totalSlewN = opsimdb.fetchTotalSlewN()

    slicerList=[]
    histNum = 0

    # Calculate the mean and median slewtime.
    metricList = []
    # Mean Slewtime
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':'Slew Summary'}))
    # Median Slewtime
    metricList.append(configureMetric('MedianMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':'Slew Summary'}))
    # Calculate the total number of visits.
    metricList.append(configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'TotalNVisits'},
                         summaryStats={'IdentityMetric':{'metricName':'Count'}},
                         displayDict={'group':'Slew Summary'}))
    # Calculate the total open shutter time.
    metricList.append(configureMetric('SumMetric', kwargs={'col':'visitExpTime', 'metricName':'Open Shutter Time'},
                         summaryStats={'IdentityMetric':{'metricName':'Time (s)'}},
                         displayDict={'group':'Slew Summary'}))
    # Number of nights
    metricList.append(configureMetric('UniqueMetric', kwargs={'col':'night'},
                                     displayDict={'group':'Slew Summary'}))
    # Mean exposure time
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'visitExpTime'},
                                      displayDict={'group':'Slew Summary'}))
    # Mean visit time
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'visitTime'},
                                      displayDict={'group':'Slew Summary'}))
    metricDict = makeDict(*metricList)
    slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], metadata='All Visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)




    # Stats for angle:
    angles = ['telAlt', 'telAz', 'rotTelPos']

    for angle in angles:
        metricList = []
        metricList.append(configureMetric('MaxMetric', kwargs={'col':angle},
                                          displayDict={'group':'Slew Angle Stats', 'subgroup':angle}))
        metricList.append(configureMetric('MinMetric', kwargs={'col':angle},
                                          displayDict={'group':'Slew Angle Stats', 'subgroup':angle}))
        metricList.append(configureMetric('MeanMetric', kwargs={'col':angle},
                                          displayDict={'group':'Slew Angle Stats', 'subgroup':angle}))
        metricList.append(configureMetric('RmsMetric', kwargs={'col':angle},
                                          displayDict={'group':'Slew Angle Stats', 'subgroup':angle}))
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], metadata=angle,
                                 metadataVerbatim=True, table='slewState')
        slicerList.append(slicer)




    # Make some calls to other tables to get slew stats
    colDict = {'domAltSpd':'Dome Alt Speed','domAzSpd':'Dome Az Speed','telAltSpd': 'Tel Alt Speed',
               'telAzSpd': 'Tel Az Speed', 'rotSpd':'Rotation Speed'}
    for key in colDict.keys():
        metricList=[]
        metricList.append(configureMetric('MaxMetric', kwargs={'col':key},
                                          displayDict={'group':'Slew Speed', 'subgroup':colDict[key]}))
        metricList.append(configureMetric('MeanMetric', kwargs={'col':key},
                                          displayDict={'group':'Slew Speed', 'subgroup':colDict[key]}))
        metricList.append(configureMetric('MaxPercentMetric', kwargs={'col':key, 'metricName':'% of slews'},
                                          displayDict={'group':'Slew Speed', 'subgroup':colDict[key]}))

        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''],
                                 table='slewMaxSpeeds', metadata=colDict[key], metadataVerbatim=True)
        slicerList.append(slicer)


    # Use the slew stats
    slewTypes = ['DomAlt', 'DomAz', 'TelAlt', 'TelAz', 'Rotator', 'Filter',
                 'TelOpticsOL', 'Readout', 'Settle', 'TelOpticsCL']

    for slewType in slewTypes:
        metricList = []
        metricList.append(configureMetric('ActivePercentMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActivePerc',
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricList.append(configureMetric('ActiveMeanMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActiveAve'},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricList.append(configureMetric('ActiveMaxMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'Max'},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )

        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0 and activity="%s"'%slewType],
                                 table='slewActivities', metadata=slewType,
                                 metadataVerbatim=True)
        slicerList.append(slicer)
        metricList = []
        metricList.append(configureMetric('ActivePercentMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActivePerc in crit',
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricList.append(configureMetric('ActiveMeanMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActiveAve in crit'},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0 and inCriticalPath="True" and activity="%s"'%slewType],
                                 table='slewActivities',
                                 metadata=slewType,
                                 metadataVerbatim=True)
        slicerList.append(slicer)



    config.slicers=makeDict(*slicerList)
    return config
