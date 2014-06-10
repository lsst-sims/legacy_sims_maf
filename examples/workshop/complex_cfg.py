# A more complex MAF configuration that uses python loops to configure things for every filter

# to run:
# runDriver.py complex_cfg.py

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict

# Set the output directory
root.outputDir = './Complex_out'
# Set the database to use (the example db included in the git repo)
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
# Name of the output table in the database
root.opsimName = 'Example'

# Make an empty list to hold all the binner configs
binList = []

# Define the filters we want to loop over
filters = ['u','g','r','i','z','y']
# Make a dict of what colors to use for different filters
colors={'u':'m','g':'b','r':'g','i':'y','z':'Orange','y':'r'}

# Resolution to use for HEALpixels
nside = 64

# Compute the coadded depth and median seeing for each filter
for filt in filters:
    metric1 = makeMetricConfig('Coaddm5Metric', params=[],
                               summaryStats={'MeanMetric':{}}, plotDict={'cbarFormat':'%.3g'})
    metric2 = makeMetricConfig('MedianMetric', params=['finSeeing'],
                               summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
    binner = makeBinnerConfig('HealpixBinner',
                              metricDict=makeDict(metric1,metric2),
                              constraints=['filter = "%s"'%filt])
    binList.append(binner)

# Now do coadd depth and median seeing, but use the hexdither positions.
# Note the addition of metricName kwargs to make each metric output unique
for filt in filters:
    metric1 = makeMetricConfig('Coaddm5Metric', params=[],
                               summaryStats={'MeanMetric':{}},
                               kwargs={'metricName':'coadd_dither'}, plotDict={'cbarFormat':'%.3g'})
    metric2 = makeMetricConfig('MedianMetric', params=['finSeeing'],
                               summaryStats={'MeanMetric':{}, 'RmsMetric':{}},
                               kwargs={'metricName':'seeing_dither'})
    binner = makeBinnerConfig('HealpixBinner',
                              metricDict=makeDict(metric1,metric2),
                              constraints=['filter = "%s"'%filt],
                              kwargs={'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'})
    binList.append(binner)




# Look at the single-visit depth and airmass for observations in each filter and merge them into a single histogram
for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['fivesigma_ps'], plotDict={'histMin':20, 'histMax':26},
                          histMerge={'histNum':1, 'legendloc':'upper right',
                                     'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'fivesigma_ps'},
                              setupKwargs={'binsize':0.1},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)]) 
    binList.append(binner)
    m1 = makeMetricConfig('CountMetric', params=['airmass'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},
                              setupKwargs={'binsize':0.05},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)])
    binList.append(binner)


# Stats on airmass and seeing for all observations:
m1 = makeMetricConfig('MeanMetric', params=['finSeeing'],
                          summaryStats={'IdentityMetric':{}})
m2 = makeMetricConfig('MeanMetric', params=['airmass'],
                          summaryStats={'IdentityMetric':{}})
m3 = makeMetricConfig('RmsMetric', params=['finSeeing'],
                          summaryStats={'IdentityMetric':{}})
m4 = makeMetricConfig('RmsMetric', params=['airmass'],
                          summaryStats={'IdentityMetric':{}})
binner = makeBinnerConfig('UniBinner', metricDict=makeDict(m1,m2,m3,m4),
                          constraints=[''])


# Run some Cadence metrics
m1 = makeMetricConfig('SupernovaMetric', kwargs={'m5col':'fivesigma_modified', 'redshift':0.1, 'resolution':5.}, plotDict={'percentileClip':95.})
m2 = makeMetricConfig('ParallaxMetric', kwargs={'metricName':'Parallax_normed', 'normalize':True})
m3 = makeMetricConfig('ParallaxMetric')
m4 = makeMetricConfig('ProperMotionMetric', plotDict={'percentileClip':95})
m5 = makeMetricConfig('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed'})
binner =  makeBinnerConfig('HealpixBinner', kwargs={"nside":nside},
                           metricDict=makeDict(m2,m3,m4,m5),
                           constraints=[''])
binList.append(binner)

# Run those same Cadence metrics on the hexdither positions
m1 = makeMetricConfig('SupernovaMetric', kwargs={'metricName':'SN_dith','m5col':'fivesigma_modified',
                                                 'redshift':0.1, 'resolution':5.},
                      plotDict={'percentileClip':95.})
m2 = makeMetricConfig('ParallaxMetric', kwargs={'metricName':'Parallax_normed_dith', 'normalize':True})
m3 = makeMetricConfig('ParallaxMetric', kwargs={'metricName':'Parallax_dith'})
m4 = makeMetricConfig('ProperMotionMetric',kwargs={'metricName':'PM_dith'},
                      plotDict={'percentileClip':95})
m5 = makeMetricConfig('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed_dith'})
binner =  makeBinnerConfig('HealpixBinner',metricDict=makeDict(m2,m3,m4,m5),
                           constraints=[''],
                           kwargs={"nside":nside,'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'})
binList.append(binner)



root.binners = makeDict(*binList)
