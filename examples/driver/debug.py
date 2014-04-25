from lsst.sims.maf.driver.mafConfig import *


# Setup Database access
root.outputDir ='./StarOut_Fields'
#root.dbAddress ={'dbAddress':'sqlite:///opsim.sqlite'}
#root.opsimNames = ['opsim']
root.dbAddress ={'dbAddress':'sqlite:///../opsim_small.sqlite'}
root.opsimNames = ['opsim_small']



filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
#filters=['r']

# 10 year Design Specs
nvisitBench={'u':56,'g':80, 'r':184, 'i':184, 'z':160, 'y':160} 
nVisits_plotRange = {'all': 
                     {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}
mag_zpoints={'u':26.1,'g':27.4, 'r':27.5, 'i':26.8, 'z':26.1, 'y':24.9}
sky_zpoints = {'u':21.8, 'g':22., 'r':21.3, 'i':20.0, 'z':19.1, 'y':17.5}
seeing_norm = {'u':0.77, 'g':0.73, 'r':0.7, 'i':0.67, 'z':0.65, 'y':0.63}

binList=[]

propids = [215, 216, 217, 218, 219]
WFDpropid = 217
DDpropid = 219


# The merged histograms for basics 
for i,f in enumerate(filters):
    m1 = makeMetricConfig('CountMetric', params=['5sigma_modified'],
                          histMerge={'histNum':1, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'5sigma_modified'},
                              metricDict=makeDict(m1), constraints=["filter = '%s'"%f])
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['perry_skybrightness'],
                          histMerge={'histNum':2, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'perry_skybrightness'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)
    
    m1 = makeMetricConfig('CountMetric', params=['finSeeing'],
                          histMerge={'histNum':3, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'finSeeing'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)

    m1 = makeMetricConfig('CountMetric', params=['airmass'],
                          histMerge={'histNum':4, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )
    binner = makeBinnerConfig('OneDBinner', kwargs={"sliceDataColName":'airmass'},
                              metricDict=makeDict(m1), constraints=["filter = '%s' "%f]) #and propID = %i"%(f,WFDpropid)])
    binList.append(binner)


root.binners=makeDict(*binList)
