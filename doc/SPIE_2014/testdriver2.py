# THIS WILL NOT RUN WITH THE SPIE branch of MAF, but it's intended to be the 'new' style "slicer" (instead of binner) format.

from lsst.sims.maf.driver.mafConfig import makeBinnerConfig, makeMetricConfig, makeDict
import lsst.sims.maf.utils as utils

# Configure the output directory.
root.outputDir = './MAFOut'

# Configure our database access information.
root.dbAddress = {'dbAddress':'sqlite:///opsim3_61.db'}
root.opsimName = 'opsim3_61'

# Some parameters to help control the plotting ranges below.
nVisits_plotRange = {'g':[50,220], 'r':[150, 310]}

# Use a MAF utility to determine the expected design and stretch goals for number of visits, etc.
design, stretch = utils.scaleStretchDesign(10)
mag_zpoints = design['coaddedDepth']

# Some parameters we're using below.
nside = 128
nvisits = 10

# A list to save the configuration parameters for our Slicers + Metrics.
configList=[]

# Loop through r and i filters, to do some simple analysis.
filters = ['r', 'i']
colors = {'r':'b', 'i':'y'}
for f in filters:
    # Configure a metric that will calculate the mean of the seeing
    #  Adding the 'IdentityMetric' to the summaryStats means it will print the output to a file.
    m1 = makeMetricConfig('MeanMetric', params=['seeing'], summaryStats={'IdentityMetric':{}})
    # Configure a metric that will calculate the rms of the seeing
    m2 = makeMetricConfig('RmsMetric', params=['seeing'], summaryStats={'IdentityMetric':{}})
    # Combine these metrics with the UniSlicer and a SQL constraint based on the filter, so
    #  that we will now calculate the mean and rms of the seeing for all r band visits
    #  (and then the mean and rms of the seeing for all i band visits).
    slicer = makeSlicerConfig('UniSlicer', metricDict=makeDict(m1, m2),
                              constraints=['filter = "%s"' %(f)])
    # Add this configured slicer (carrying the metric information and the sql constraint) into a list.
    configList.append(slicer)
    # Configure a metric + a OneDSlicer so that we can count how many visits 
    #  are within in each interval of the seeing value in the OneDSlicer. 
    m1 = makeMetricConfig('CountMetric', params=['Airmass'], kwargs={'metricName':'Airmass'},
                          # Set up a additional histogram so that the outputs of these count metrics in each
                          #   filter get combined into a single plot (with both r and i band). 
                          histMerge = {'histNum':1, 'legendloc':'upper right', 'label':'%s band' %(f),
                                       'xlabel':'Airmass', 'color':colors[f]})
    # Set up the OneDSlicer, including setting the interval size for slicing.
    slicer = makeSlicerConfig('OneDSlicer', kwargs={'sliceColName':'Airmass', 'slicesize':0.02},
                              metricDict=makeDict(m1), constraints=['filter = "%s"' %(f)])
    configList.append(slicer)



# Loop through the different dither options.
dithlabels = [' No dithering', ' Random dithering', ' Hex dithering']
slicerNames = ['HealpixSlicer' ,'HealpixSlicerRandom', 'HealpixSlicerDither'] 
for dithlabel, slicerName in zip(dithlabels, slicerNames):
    # Set up parameters for slicer, depending on what dither pattern we're using.
    if slicerName == 'HealpixSlicer':
        slicerName = 'HealpixSlicer'
        slicerkwargs = {'nside':nside}
        slicermetadata = 'No dither'
    elif slicerName == 'HealpixSlicerDither':
        slicerName = 'HealpixSlicer'
        slicerkwargs = {'nside':nside, 'spatialkey1':'hexdithra', 'spatialkey2':'hexdithdec'}
        slicermetadata = 'Hex dither'
    elif slicerName == 'HealpixSlicerRandom':
        slicerName = 'HealpixSlicer'
        slicerkwargs = {'nside':nside, 'spatialkey1':'randomRADither', 'spatialkey2':'randomDecDither'}
        slicermetadata = 'Random dither'
    # Configure QuickRevisit metric to count number times we have more than X visits within a night.
    m1 = makeMetricConfig('QuickRevisitMetric', kwargs={'nVisitsInNight':nvisits}, 
                        plotDict={'plotMin':0, 'plotMax':20, 'histMin':0, 'histMax':100},
                        summaryStats={'MeanMetric':{}},
                        # Add it to a 'merged' histogram, which will combine metric values from
                        #  all dither patterns.
                        histMerge={'histNum':2, 'legendloc':'upper right', 'label':dithlabel,
                                    'histMin':0, 'histMax':50,
                                    'xlabel':'Number of Nights with more than %d visits' %(nvisits),
                                    'bins':50})
    # Configure Proper motion metric to analyze expected proper motion accuracy.
    m2 = makeMetricConfig('ProperMotionMetric', kwargs={'m5Col':'5sigma_modified', 'seeingCol':'seeing',
                                                        'metricName':'Proper Motion @20 mag'},
                        plotDict={'percentileClip':95, 'plotMin':0, 'plotMax':2.0},
                        histMerge={'histNum':3, 'legendloc':'upper right', 'label':dithlabel,
                                   'histMin':0, 'histMax':1.5})
    # Configure another proper motion metric where input star is r=24 rather than r=20.
    m3 = makeMetricConfig('ProperMotionMetric', kwargs={'m5Col':'5sigma_modified', 'seeingCol':'seeing',
                                                        'rmag':24, 'metricName':'Proper Motion @24 mag'},
                        plotDict={'percentileClip':95, 'plotMin':0, 'plotMax':2.0},
                        histMerge={'histNum':4, 'legendloc':'upper right', 'label':dithlabel,
                                   'histMin':0, 'histMax':1.5})
    # Configure a Healpix slicer that uses either the opsim original pointing, or one of the dithered RA/Dec values.
    slicer = makeSlicerConfig(slicerName, kwargs=slicerkwargs, 
                          metricDict=makeDict(m1, m2, m3), constraints=[''], metadata = dithlabel)
    # Add this configured slicer (which carries the metrics and sql constraints with it) to our list.
    configList.append(slicer)

    # Loop through filters g and r and calculate number of visits and coadded depth in these filters
    for f in ['g', 'r']:
        # Reset histNum to be 5 or 6 depending on filter
        #   (so same filter, different dithered slicers end in same merged histogram)
        if f == 'g':
            histNum = 5
        elif f == 'r':
            histNum = 7
        # Configure a metric to count the number of visits in this band.
        m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits %s band' %(f)},
                              plotDict={'units':'Number of Visits', 'cbarFormat':'%d',
                                      'plotMin':nVisits_plotRange[f][0],
                                      'plotMax':nVisits_plotRange[f][1],
                                      'histMin':nVisits_plotRange[f][0],
                                      'histMax':nVisits_plotRange[f][1]},
                                summaryStats={'MeanMetric':{}, 'RmsMetric':{}},
                                histMerge={'histNum':histNum, 'legendloc':'upper right',
                                           'histMin':nVisits_plotRange[f][0],
                                           'histMax':nVisits_plotRange[f][1],
                                           'label':dithlabel,
                                           'bins':(nVisits_plotRange[f][1]-nVisits_plotRange[f][0])})
        histNum += 1
        # Configure a metric to count the coadded m5 depth in this band.
        m2 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'5sigma_modified',
                                                       'metricName':'Coadded m5 %s band' %(f)},
                                plotDict={'zp':mag_zpoints[f],
                                          'percentileClip':95., 'plotMin':-1.5, 'plotMax':0.75,
                                          'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]},
                                summaryStats={'MeanMetric':{}, 'RmsMetric':{}},
                                histMerge={'histNum':histNum, 'legendloc':'upper left', 'bins':150,
                                            'label':dithlabel, 'histMin':24.5, 'histMax':28.5})
        # Configure the slicer for these two metrics
        #  (separate from healpix slicer above because of filter constraint).
        slicer = makeSlicerConfig(slicerName, kwargs=slicerkwargs, 
                            metricDict=makeDict(m1, m2), constraints=['filter="%s"'%(f)],
                            metadata = dithlabel)
        configList.append(slicer)



root.slicers=makeDict(*configList)
