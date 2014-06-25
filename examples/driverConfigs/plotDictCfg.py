# Examples of all the plotting dictionary options

from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict

root.outputDir = './PlotDict'
root.dbAddress = {'dbAddress':'sqlite:///opsimblitz2_1060_sqlite.db'}
root.opsimName =  'ob2_1060'
nside=16
slicerList=[]

plotDict={ 'units':'units!', 'title':'title!', 'cmap':'RdBu', 
           'plotMin':0.6, 'plotMax':1.5, 'histMin':0.5, 'histMax':1.7, 'ylabel':'ylabel!',
           'label':'label!', 'addLegend':True, 'color':'r', 'bins':150, 'cbarFormat':'%.4g'}


m1 = configureMetric('MeanMetric', params=['finSeeing'], plotDict=plotDict)
m2 = configureMetric('MeanMetric', params=['finSeeing'], kwargs={'metricName':'mean_default'})

metricDict = makeDict(m1,m2)

slicer = configureSlicer('HealpixSlicer',kwargs={"nside":nside},
                          metricDict = metricDict,constraints=['filter="r"'])
# For skymap units, title, ylog, cbarFormat, cmap, percentileClip, plotMin, plotMax, zp, normVal
# also on hist:  ylabel, bins, cumulative, histMin, histMax, ylog, color, scale, label, addLegend

slicerList.append(slicer)

plotDict={ 'units':'units!', 'title':'title!', 'histMin':0.5, 'histMax':1.7, 'ylabel':'ylabel!',
           'xlabel':'xlabel!', 'label':'label!', 'addLegend':True, 'color':'r', 'yMin':0.,'yMax':10000.}


m1 = configureMetric('CountMetric', params=['finSeeing'], kwargs={'metricName':'wplotdict'},plotDict=plotDict)
m2 = configureMetric('CountMetric', params=['finSeeing'])

metricDict = makeDict(m1,m2)
slicer = configureSlicer('OneDSlicer', kwargs={'sliceColName':'finSeeing'},
                          metricDict = metricDict,constraints=['filter="r"'])
slicerList.append(slicer)

#for OneD:  title, units, label, addLegend, legendloc, filled, alpha, ylog, ylabel, xlabel, yMin, yMax, histMin, histMax




root.slicers=makeDict(*slicerList)
