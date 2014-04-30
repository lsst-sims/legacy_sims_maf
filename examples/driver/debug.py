# A MAF config that replicates the SSTAR plots
from lsst.sims.maf.driver.mafConfig import *

# Setup Database access
root.outputDir ='./StarOut_Fields'

small = True # Use the small database included in the repo

if small:
    root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
    root.opsimNames = ['opsim_small']
    propids = [186,187,188,189]
    WFDpropid = 188
    DDpropid = 189 #?
else:
    root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
    root.opsimNames = ['opsim']
    propids = [215, 216, 217, 218, 219]
    WFDpropid = 217
    DDpropid = 219


#root.dbAddress ={'dbAddress':'mysql://www:zxcvbnm@localhost/OpsimDB'}
#root.opsimNames = ['output_opsimblitz2_1020']
#propids = [93,94,95,96,97,98]
#WFDpropid = 96 #XXX guessing here...
#DDpropid = 94 #XXX Guess




filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'Orange','y':'r'}
#filters=['r']

# 10 year Design Specs
nvisitBench={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} 
nVisits_plotRange = {'all': 
                     #{'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     {'u':[0, 200], 'g':[0,200], 'r':[0, 200], 'i':[0, 200], 'z':[0, 200], 'y':[0,200]},
                     'visits':
                         {'u':[0, 70], 'g':[0,100], 'r':[0, 250], 'i':[0, 250], 'z':[0, 200], 'y':[0,200]  }, 
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}
mag_zpoints={'u':26.1,'g':27.4, 'r':27.5, 'i':26.8, 'z':26.1, 'y':24.9}
sky_zpoints = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
seeing_norm = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

binList=[]


# Metrics per filter, WFD only
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'percentileClip':75., 'units':'Number of Visits', 'label':f, 'addLegend':True,
                                    'histMin':nVisits_plotRange['all'][f][0], 'histMax':nVisits_plotRange['all'][f][1]},
                          histMerge={'histNum':5, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitBench[f], 'percentileClip':80., 'units':'Number of Visits/Benchmark (%d)' %(nvisitBench[f])})
    m3 = makeMetricConfig('MedianMetric', params=['5sigma_ps'])
    m4 = makeMetricConfig('Coaddm5Metric', kwargs={'m5col':'5sigma_ps', 'metricName':'Coaddm5WFD'},
                          plotDict={'zp':mag_zpoints[f], 'plotMin':-0.8, 'plotMax':0.8, 'percentileClip':95.,
                                    'units':'Co-add (m5 - %.1f)'%mag_zpoints[f], 'histMin':23, 'histMax':28},
                           histMerge={'histNum':7, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f})             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_unit':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_unit':'X'})
    metricDict = makeDict(m1)
    binner = makeBinnerConfig('OpsimFieldBinner', metricDict=metricDict, constraints=["filter = \'%s\' and propID = %d"%(f, WFDpropid)])
    binList.append(binner)


                            
        

root.binners=makeDict(*binList)


