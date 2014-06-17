# Test out new cadence metrics
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict
import lsst.sims.maf.utils as utils

root.outputDir ='./Cadence'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}#, 'OutputTable':'output'}
root.opsimName = 'ob2_1060'

# Connect to the database to fetch some values we're using to help configure the driver.                                                             
opsimdb = utils.connectOpsimDb(root.dbAddress)

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

filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
filters=['r']

slicerList=[]
nside=64
leafsize = 100 # For KD-tree


########### Early Seeing Metrics ################
seeing_limit = 0.7 # Demand seeing better than this
for f in filters:
    m1 = configureMetric('BinaryMetric', params=['finSeeing'], summaryStats={'SumMetric':{}})
    slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside},metricDict=makeDict(m1),
                              constraints=['night < 365 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'night < 730 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'filter = "%s" and finSeeing < %s'%(f,seeing_limit)],
                              setupKwargs={"leafsize":leafsize})
    slicerList.append(slicer)

# Look at the minimum seeing per field, and the fraction of observations below the "good" limit
for f in filters:
    m1 = configureMetric('TemplateExistsMetric')
    m2 = configureMetric('MinMetric', params=['finSeeing'])
    m3 = configureMetric('FracBelowMetric', params=['finSeeing'], kwargs={'cutoff':seeing_limit})
    slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside},metricDict=makeDict(m1,m2,m3),
                              constraints=['night < 365 and filter = "%s"'%(f),
                                           'night < 730 and filter = "%s"'%(f),
                                           'filter = "%s"'%(f)],
                              setupKwargs={"leafsize":leafsize})
    slicerList.append(slicer)


#########  Supernova Metric ############
m1 = configureMetric('SupernovaMetric', kwargs={'m5col':'fivesigma_modified', 'redshift':0.1, 'resolution':5.}, plotDict={'percentileClip':95.})
########   Parallax and Proper Motion ########
m2 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_normed', 'normalize':True})
m3 = configureMetric('ParallaxMetric')
m4 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_24', 'rmag':24})
m5 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_24_normed', 'rmag':24, 'normalize':True})
m6 = configureMetric('ProperMotionMetric', plotDict={'percentileClip':95})
m7 = configureMetric('ProperMotionMetric', kwargs={'rmag':24, 'metricName':'PM_24'}, plotDict={'percentileClip':95})
m8 = configureMetric('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed'})
m9 = configureMetric('ProperMotionMetric', kwargs={'rmag':24,'normalize':True, 'metricName':'PM_24_normed'})
slicer =  configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                           metricDict=makeDict(m1,m2,m3,m4,m5),
                           constraints=['night < 365', ''], setupKwargs={"leafsize":leafsize})
slicerList.append(slicer)


########### Time Uniformity Metric ###########
constraints=[]
for f in filters:
    constraints.append('filter = "%s"'%f)
constraints.append('')
constraints.append('night < 365')
m1 = configureMetric('UniformityMetric', plotDict={'plotMin':0., 'plotMax':1.})
slicer = configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                           metricDict=makeDict(m1),
                           constraints=constraints, setupKwargs={"leafsize":leafsize})
slicerList.append(slicer)



root.slicers = makeDict(*slicerList)

