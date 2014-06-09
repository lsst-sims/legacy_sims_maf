# Test out new cadence metrics
from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict
import lsst.sims.maf.utils as utils
  



root.outputDir ='./Cadence'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}#, 'OutputTable':'output'}
root.opsimName = 'example'

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

binList=[]
nside=64
leafsize = 100 # For KD-tree


########### Early Seeing Metrics ################
seeing_limit = 0.7 # Demand seeing better than this
for f in filters:
    m1 = makeMetricConfig('BinaryMetric', params=['finSeeing'], summaryStats={'SumMetric':{}})
    binner = makeBinnerConfig('HealpixBinner',kwargs={"nside":nside},metricDict=makeDict(m1),
                              constraints=['night < 365 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'night < 730 and filter = "%s" and finSeeing < %s'%(f,seeing_limit),
                                           'filter = "%s" and finSeeing < %s'%(f,seeing_limit)],
                              setupKwargs={"leafsize":leafsize})
    binList.append(binner)

# Look at the minimum seeing per field, and the fraction of observations below the "good" limit
for f in filters:
    m1 = makeMetricConfig('TemplateExistsMetric')
    m2 = makeMetricConfig('MinMetric', params=['finSeeing'])
    m3 = makeMetricConfig('FracBelowMetric', params=['finSeeing'], kwargs={'cutoff':seeing_limit})
    binner = makeBinnerConfig('HealpixBinner',kwargs={"nside":nside},metricDict=makeDict(m1,m2,m3),
                              constraints=['night < 365 and filter = "%s"'%(f),
                                           'night < 730 and filter = "%s"'%(f),
                                           'filter = "%s"'%(f)],
                              setupKwargs={"leafsize":leafsize})
    binList.append(binner)


#########  Supernova Metric ############
m1 = makeMetricConfig('SupernovaMetric', kwargs={'m5col':'fivesigma_modified', 'redshift':0.1, 'resolution':5.}, plotDict={'percentileClip':95.})
########   Parallax and Proper Motion ########
m2 = makeMetricConfig('ParallaxMetric', kwargs={'metricName':'Parallax_normed', 'normalize':True})
m3 = makeMetricConfig('ParallaxMetric')
m4 = makeMetricConfig('ProperMotionMetric', plotDict={'percentileClip':95})
m5 = makeMetricConfig('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed'})
binner =  makeBinnerConfig('HealpixBinner', kwargs={"nside":nside},
                           metricDict=makeDict(m1,m2,m3,m4,m5),
                           constraints=['night < 365', ''], setupKwargs={"leafsize":leafsize})
binList.append(binner)


########### Time Uniformity Metric ###########
constraints=[]
for f in filters:
    constraints.append('filter = "%s"'%f)
constraints.append('')
m1 = makeMetricConfig('UniformityMetric', plotDict={'plotMin':0., 'plotMax':1.})
binner = makeBinnerConfig('HealpixBinner', kwargs={"nside":nside},
                           metricDict=makeDict(m1),
                           constraints=constraints, setupKwargs={"leafsize":leafsize})
binList.append(binner)



root.binners = makeDict(*binList)

