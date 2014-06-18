# A more complex MAF configuration that uses python loops to configure things for every filter

# to run:
# runDriver.py complex_cfg.py

# Import MAF helper functions 
from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

# Set the output directory
root.outputDir = './Complex_out'
# Set the database to use (the example db included in the git repo)
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
# Name of the output table in the database
root.opsimName = 'Example'

# Make an empty list to hold all the slicer configs
binList = []

# Define the filters we want to loop over
filters = ['u','g','r','i','z','y']
# Make a dict of what colors to use for different filters
colors={'u':'m','g':'b','r':'g','i':'y','z':'Orange','y':'r'}

# Resolution to use for HEALpixels
nside = 64

# Compute the coadded depth and median seeing for each filter
for filt in filters:
    metric1 = configureMetric('Coaddm5Metric', params=[],
                               summaryStats={'MeanMetric':{}}, plotDict={'cbarFormat':'%.3g'})
    metric2 = configureMetric('MedianMetric', params=['finSeeing'],
                               summaryStats={'MeanMetric':{}, 'RmsMetric':{}})
    slicer = configureSlicer('HealpixSlicer',
                              metricDict=makeDict(metric1,metric2),
                              constraints=['filter = "%s"'%filt])
    binList.append(slicer)

# Now do coadd depth and median seeing, but use the hexdither positions.
# Note the addition of metricName kwargs to make each metric output unique
for filt in filters:
    metric1 = configureMetric('Coaddm5Metric', params=[],
                               summaryStats={'MeanMetric':{}},
                               kwargs={'metricName':'coadd_dither'}, plotDict={'cbarFormat':'%.3g'})
    metric2 = configureMetric('MedianMetric', params=['finSeeing'],
                               summaryStats={'MeanMetric':{}, 'RmsMetric':{}},
                               kwargs={'metricName':'seeing_dither'})
    slicer = configureSlicer('HealpixSlicer',
                              metricDict=makeDict(metric1,metric2),
                              constraints=['filter = "%s"'%filt],
                              kwargs={'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'})
    binList.append(slicer)




# Look at the single-visit depth and airmass for observations in each filter and merge them into a single histogram
for f in filters:
    m1 = configureMetric('CountMetric', params=['fivesigma_ps'], plotDict={'histMin':20, 'histMax':26},
                          histMerge={'histNum':1, 'legendloc':'upper right',
                                     'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceDim":'fivesigma_ps','binsize':0.1,},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)]) 
    binList.append(slicer)
    m1 = configureMetric('CountMetric', params=['airmass'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    slicer = configureSlicer('OneDSlicer', kwargs={"sliceDim":'airmass','binsize':0.05},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%(f)])
    binList.append(slicer)


# Stats on airmass and seeing for all observations:
m1 = configureMetric('MeanMetric', params=['finSeeing'],
                          summaryStats={'IdentityMetric':{}})
m2 = configureMetric('MeanMetric', params=['airmass'],
                          summaryStats={'IdentityMetric':{}})
m3 = configureMetric('RmsMetric', params=['finSeeing'],
                          summaryStats={'IdentityMetric':{}})
m4 = configureMetric('RmsMetric', params=['airmass'],
                          summaryStats={'IdentityMetric':{}})
slicer = configureSlicer('UniSlicer', metricDict=makeDict(m1,m2,m3,m4),
                          constraints=[''])


# Run some Cadence metrics
m1 = configureMetric('SupernovaMetric', kwargs={'m5col':'fivesigma_modified', 'redshift':0.1, 'resolution':5.}, plotDict={'percentileClip':95.})
m2 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_normed', 'normalize':True})
m3 = configureMetric('ParallaxMetric')
m4 = configureMetric('ProperMotionMetric', plotDict={'percentileClip':95})
m5 = configureMetric('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed'})
slicer =  configureSlicer('HealpixSlicer', kwargs={"nside":nside},
                           metricDict=makeDict(m2,m3,m4,m5),
                           constraints=[''])
binList.append(slicer)

# Run those same Cadence metrics on the hexdither positions
m1 = configureMetric('SupernovaMetric', kwargs={'metricName':'SN_dith','m5col':'fivesigma_modified',
                                                 'redshift':0.1, 'resolution':5.},
                      plotDict={'percentileClip':95.})
m2 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_normed_dith', 'normalize':True})
m3 = configureMetric('ParallaxMetric', kwargs={'metricName':'Parallax_dith'})
m4 = configureMetric('ProperMotionMetric',kwargs={'metricName':'PM_dith'},
                      plotDict={'percentileClip':95})
m5 = configureMetric('ProperMotionMetric', kwargs={'normalize':True, 'metricName':'PM_normed_dith'})
slicer =  configureSlicer('HealpixSlicer',metricDict=makeDict(m2,m3,m4,m5),
                           constraints=[''],
                           kwargs={"nside":nside,'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'})
binList.append(slicer)



root.slicers = makeDict(*binList)
