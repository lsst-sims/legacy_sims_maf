# Run new cadence metrics.

import os
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils


def mConfig(config, runName, dbDir='.', outputDir='Cadence', **kwargs):
    """
    A MAF config to run the cadence metrics on an OpSim run.
    """

    # Setup Database access
    config.outputDir = outputDir
    sqlitefile = os.path.join(dbDir, runName + '_sqlite.db')
    config.dbAddress ={'dbAddress':'sqlite:///'+sqlitefile}
    config.opsimName = runName

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


    # Fetch the total number of visits (to create fraction)
    totalNVisits = opsimdb.fetchNVisits()

    allfilters = ['u','g','r','i','z','y']
    colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
    filtorder = {'u':1,'g':2,'r':3,'i':4,'z':5,'y':6}

    usefilters=allfilters#['r']

    slicerList=[]
    nside=128

    ########### Early Seeing Metrics ################
    seeing_limit = 0.7 # Demand seeing better than this
    for f in usefilters:
        m1 = configureMetric('BinaryMetric', kwargs={'col':'finSeeing'},
                             summaryStats={'SumMetric':{}},
                             displayDict={'group':'Cadence', 'subgroup':'Early Good Seeing',
                                          'order':filtorder[f],
                                          'caption':'Points where there are visits with seeing better than %.1f, for visits matching the sql constraints.' %(seeing_limit)})
        slicer = configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                                 metricDict=makeDict(m1),
                                 constraints=
                                 ['night < 365 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                  'night < 730 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                  'filter = "%s" and finSeeing < %s'%(f, seeing_limit)])
        slicerList.append(slicer)

    # Look at the minimum seeing per field, and the fraction of observations below the "good" limit
    for f in usefilters:
        m1 = configureMetric('TemplateExistsMetric', displayDict={'group':'Cadence',
                                                                  'subgroup':'Early Good Seeing',
                                                                  'order':filtorder[f]})
        m2 = configureMetric('MinMetric', kwargs={'col':'finSeeing'},
                             displayDict={'group':'Cadence', 'subgroup':'Early Good Seeing',
                                          'order':filtorder[f]})
        m3 = configureMetric('FracBelowMetric', kwargs={'col':'finSeeing', 'cutoff':seeing_limit},
                             displayDict={'group':'Cadence', 'subgroup':'Early Good Seeing',
                                          'order':filtorder[f]})
        slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside}, metricDict=makeDict(m1,m2,m3),
                                constraints=['night < 365 and filter = "%s"'%(f),
                                            'night < 730 and filter = "%s"'%(f),
                                            'filter = "%s"'%(f)])
        slicerList.append(slicer)


    #########  Supernova Metric ############
    m1 = configureMetric('SupernovaMetric',
                         kwargs={'m5Col':'fiveSigmaDepth', 'redshift':0.1, 'resolution':5.},
                         plotDict={'percentileClip':95.},
                         displayDict={'group':'Cadence', 'subgroup':'Supernova'})
    slicer =  configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                            metricDict=makeDict(m1),
                            constraints=['night < 365', ''])
    #slicerList.append(slicer)

    ########   Parallax and Proper Motion ########
    m2 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax Normed', 'normalize':True},
                         displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m3 = configureMetric('ParallaxMetric', displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m4 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax 24mag', 'rmag':24},
                         displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m5 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax 24mag Normed',
                                                   'rmag':24, 'normalize':True},
                        displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m6 = configureMetric('ProperMotionMetric', plotDict={'percentileClip':95},
                         displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m7 = configureMetric('ProperMotionMetric', kwargs={'rmag':24, 'metricName':'Proper Motion 24mag'},
                         plotDict={'percentileClip':95},
                         displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m8 = configureMetric('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'Proper Motion Normed'},
                         displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    m9 = configureMetric('ProperMotionMetric', kwargs={'rmag':24,'normalize':True,
                                                       'metricName':'Proper Motion 24mag Normed'},
                        displayDict={'group':'Cadence', 'subgroup':'Calibration'})
    slicer =  configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                            metricDict=makeDict(m2,m3,m4,m5,m6,m7,m8,m9),
                            constraints=['night < 365', ''])
    slicerList.append(slicer)


    ########### Time Uniformity Metric ###########
    constraints=[]
    for f in usefilters:
        constraints.append('filter = "%s"'%f)
    constraints.append('')
    constraints.append('night < 365')
    m1 = configureMetric('UniformityMetric', plotDict={'colorMin':0., 'colorMax':1.},
                         displayDict={'group':'Cadence', 'subgroup':'Uniformity'})
    slicer = configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                                metricDict=makeDict(m1),
                                constraints=constraints)
    slicerList.append(slicer)


    #### Visit Group Metric and AGN gap ######
    m1 = configureMetric('VisitGroupsMetric',
                         kwargs={'minNVisits':2, 'metricName':'VisitGroups2'},
                         plotDict={'percentile':90},
                         displayDict={'group':'Cadence', 'subgroup':'Visit Groups'})
    m2 = configureMetric('VisitGroupsMetric',
                         kwargs={'minNVisits':4, 'metricName':'VisitGroups4'},
                         plotDict={'percentile':90},
                         displayDict={'group':'Cadence', 'subgroup':'Visit Groups'})
    m3 = configureMetric('LongGapAGNMetric', displayDict={'group':'AGN'})
    slicer = configureSlicer('HealpixSlicer', kwargs={'nside':nside},
                            metricDict=makeDict(m1, m2, m3),
                             constraints=['(filter = "r") or (filter="g") or (filter="i")'])
    # These are making giant .npz files!
    #slicerList.append(slicer)



    config.slicers = makeDict(*slicerList)
    return config
