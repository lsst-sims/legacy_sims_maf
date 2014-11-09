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

    #### Set up parameters for configuring plotting dictionaries and identifying WFD proposals.

    # Connect to the database to fetch some values we're using to help configure the driver.
    opsimdb = utils.connectOpsimDb(config.dbAddress)

    # Fetch the proposal ID values from the database
    propids, WFDpropid, DDpropid, propID2Name = opsimdb.fetchPropIDs()

    # Construct a WFD SQL where clause so multiple propIDs can query by WFD:
    wfdWhere = ''
    if len(WFDpropid) == 1:
        wfdWhere = "propID = %d" %(WFDpropid[0])
    else:
        for i,propid in enumerate(WFDpropid):
            if i == 0:
                wfdWhere = wfdWhere+'('+'propID = %d ' %(propid)
            else:
                wfdWhere = wfdWhere+'or propID = %d ' %(propid)
        wfdWhere = wfdWhere+')'


    # Fetch the total number of visits (to create fraction for number of visits per proposal)
    totalNVisits = opsimdb.fetchNVisits()
    totalSlewN = opsimdb.fetchTotalSlewN()

    # Filter list, and map of colors (for plots) to filters.
    filters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    # Set up benchmark values for Stretch and Design, scaled to length of opsim run.
    runLength = opsimdb.fetchRunLength()
    design, stretch = utils.scaleStretchDesign(runLength)

    # Set zeropoints and normalization values for plots (and range for nvisits plots).
    if benchmark == 'stretch':
        sky_zpoints = stretch['skybrightness']
        seeing_norm = stretch['seeing']
        mag_zpoints = stretch['coaddedDepth']
        nvisitBench = stretch['nvisits']
    else:
        sky_zpoints = design['skybrightness']
        seeing_norm = design['seeing']
        mag_zpoints = design['coaddedDepth']
        nvisitBench = design['nvisits']
    # make sure nvisitBench not zero
    for key in nvisitBench.keys():
        if nvisitBench[key] == 0:
            print 'Changing nvisit benchmark value to not be zero.'
            nvisitBench[key] = 1

    # Set range of values for visits plots.
    nVisits_plotRange = {'all':
                         {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200],
                          'z':[100, 250], 'y':[100,250]},
                         'DD':
                         {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],
                          'z':[7000, 10000], 'y':[5000, 8000]}}

    # Set slicer name and kwargs, so that we can vary these from the command line.
    slicerNames = ['HealpixSlicer', 'HealpixSlicerDither', 'OpsimFieldSlicer']
    if slicerName == 'HealpixSlicer':
        slicerName = 'HealpixSlicer'
        nside = 128
        slicerkwargs = {'nside':nside}
        slicermetadata = ''
    elif slicerName == 'HealpixSlicerDither':
        slicerName = 'HealpixSlicer'
        nside = 128
        slicerkwargs = {'nside':nside, 'spatialkey1':'ditheredRA', 'spatialkey2':'ditheredDec'}
        slicermetadata = ' dithered'
    elif slicerName == 'OpsimFieldSlicer':
        slicerName = 'OpsimFieldSlicer'
        slicerkwargs = {}
        slicermetadata = ''
    else:
        raise ValueError('Do not understand slicerName %s: looking for one of %s' %(slicerName, slicerNames))
    print 'Using slicer %s for generic metrics over the sky.' %(slicerName)

    ####

    # Configure some standard summary statistics dictionaries to apply to appropriate metrics.

    # Note there's a complication here, you can't configure multiple versions of a summary metric since that makes a
    # dict with repeated keys.  One kinda workaround is to add blank space (or even more words) to one of
    # the keys that gets stripped out when the object is instatiated.
    standardStats={'MeanMetric':{}, 'RmsMetric':{}, 'MedianMetric':{}, 'CountMetric':{},
                   'NoutliersNsigma 1':{'metricName':'p3Sigma', 'nSigma':3.},
                   'NoutliersNsigma 2':{'metricName':'m3Sigma', 'nSigma':-3.}}
    percentileStats={'PercentileMetric 1':{'metricName':'25th%ile', 'percentile':25},
                     'PercentileMetric 2':{'metricName':'50th%ile', 'percentile':50},
                     'PercentileMetric 3':{'metricName':'75th%ile', 'percentile':75}}
    allStats = standardStats.copy()
    allStats.update(percentileStats)
    ####
    # Start specifying metrics and slicers for MAF to run.

    slicerList=[]
    histNum = 0




    # Some other summary statistics over all filters and all proposals.
    # Calculate the mean and median slewtime.
    metricList = []
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':'Slew', 'subgroup':'Slew'}))
    metricList.append(configureMetric('MedianMetric', kwargs={'col':'slewTime'},
                         displayDict={'group':'Slew', 'subgroup':'Slew'}))
    # Calculate the total number of visits.
    metricList.append(configureMetric('CountMetric', kwargs={'col':'expMJD', 'metricName':'TotalNVisits'},
                         summaryStats={'IdentityMetric':{'metricName':'Count'}},
                         displayDict={'group':'1: Summary', 'subgroup':'NVisits', 'order':0}))
    # Calculate the total open shutter time.
    metricList.append(configureMetric('SumMetric', kwargs={'col':'visitExpTime', 'metricName':'Open Shutter Time'},
                         summaryStats={'IdentityMetric':{'metricName':'Time (s)'}},
                         displayDict={'group':'1: Summary', 'subgroup':'On-sky Time'}))
    # Number of nights
    metricList.append(configureMetric('UniqueMetric', kwargs={'col':'night'},
                                     displayDict={'group':'Slew', 'subgroup':'Nights'}))
    # Slew stats
    metricList.append(configureMetric('MaxMetric', kwargs={'col':'altitude'},
                                      displayDict={'group':'Slew', 'subgroup':'Alt'}))
    metricList.append(configureMetric('MinMetric', kwargs={'col':'altitude'},
                                      displayDict={'group':'Slew', 'subgroup':'Alt'}))
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'altitude'},
                                      displayDict={'group':'Slew', 'subgroup':'Alt'}))
    metricList.append(configureMetric('RmsMetric', kwargs={'col':'altitude'},
                                      displayDict={'group':'Slew', 'subgroup':'Alt'}))
    metricList.append(configureMetric('MaxMetric', kwargs={'col':'azimuth'},
                                      displayDict={'group':'Slew', 'subgroup':'Az'}))
    metricList.append(configureMetric('MinMetric', kwargs={'col':'azimuth'},
                                      displayDict={'group':'Slew', 'subgroup':'Az'}))
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'azimuth'},
                                      displayDict={'group':'Slew', 'subgroup':'Az'}))
    metricList.append(configureMetric('RmsMetric', kwargs={'col':'azimuth'},
                                      displayDict={'group':'Slew', 'subgroup':'Az'}))
    # Mean exposure time
    metricList.append(configureMetric('MeanMetric', kwargs={'col':'visitExpTime'},
                                      displayDict={'group':'Slew', 'subgroup':'Exptime'}))

    metricDict = makeDict(*metricList)
    slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], metadata='All Visits',
                             metadataVerbatim=True)
    slicerList.append(slicer)



    # Make some calls to other tables to get slew stats
    colDict = {'domAltSpd':'Dome Alt Speed','domAzSpd':'Dome Az Speed','telAltSpd': 'Tel Alt Speed',
               'rotSpd':'Rotation Speed'}
    metricList=[]
    for key in colDict.keys():
        metricList.append(configureMetric('MaxMetric', kwargs={'col':key},
                                          displayDict={'group':'Slew', 'subgroup':colDict[key]}))
        metricList.append(configureMetric('MeanMetric', kwargs={'col':key},
                                          displayDict={'group':'Slew', 'subgroup':colDict[key]}))
        metricList.append(configureMetric('MaxPercentMetric', kwargs={'col':key},
                                          displayDict={'group':'Slew', 'subgroup':colDict[key]}))

    metricDict = makeDict(*metricList)
    slicer = configureSlicer('UniSlicer', metricDict=metricDict, constraints=[''], table='slewMaxSpeeds')
    #slicerList.append(slicer)


    # Use the slew stats
    slewTypes = ['DomAlt', 'DomAz', 'TelAlt', 'TelAz', 'Rotator', 'Filter',
                 'TelOpticsOL', 'Readout', 'Settle', 'TelOpticsCL']
    for slewType in slewTypes:
        metricList = []
        metricList.append(configureMetric('MeanMetric',
                                          kwargs={'col':'actDelay' ,'metricName':'Mean %s'%slewType},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricList.append(configureMetric('MaxMetric',
                                          kwargs={'col':'actDelay', 'metricName':'Max %s'%slewType},
                                          displayDict={'group':'Slew', 'subgroup':slewType}) )
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['activity = "%s" and actDelay>0'%slewType],
                                 table='slewActivities')
        slicerList.append(slicer)
        metricList=[]
        metricList.append(configureMetric('MeanMetric',
                                          kwargs={'col':'actDelay' ,'metricName':'Mean %s, in crit'%slewType},
                                          displayDict={'group':'Slew', 'subgroup':slewType+' in crit path'}) )
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['activity = "%s" and actDelay>0 and inCriticalPath="True"'%slewType],
                                 table='slewActivities')
        slicerList.append(slicer)


    for slewType in slewTypes:
        metricList = []
        metricList.append(configureMetric('ActivePercentMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActivePerc%s'%slewType,
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType+'active'}) )
        metricList.append(configureMetric('ActiveMeanMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActiveAve%s'%slewType,
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType+'active'}) )
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0'], table='slewActivities')
        slicerList.append(slicer)

        metricList = []
        metricList.append(configureMetric('ActivePercentMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActivePerc%s in crit'%slewType,
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType+'active in crit'}) )
        metricList.append(configureMetric('ActiveMeanMetric',
                                          kwargs={'col':'actDelay', 'activity':slewType,
                                                  'metricName':'ActiveAve%s in crit'%slewType,
                                                  'norm':100./totalSlewN},
                                          displayDict={'group':'Slew', 'subgroup':slewType+'active in crit'}) )
        metricDict = makeDict(*metricList)
        slicer = configureSlicer('UniSlicer', metricDict=metricDict,
                                 constraints=['actDelay>0 and inCriticalPath="True"'], table='slewActivities')
        slicerList.append(slicer)



    config.slicers=makeDict(*slicerList)
    return config
