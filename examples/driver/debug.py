# A MAF config that replicates the SSTAR plots

import numpy as np
from lsst.sims.maf.driver.mafConfig import *
from lsst.sims.maf.utils import runInfo

# Setup Database access
root.outputDir ='./Debug'
root.dbAddress ={'dbAddress':'sqlite:///hewelhog_1016_sqlite.db', 'fieldTable':'Field', 'sessionID':'1016', 'proposalTable': 'Proposal_Field'}
root.opsimNames = ['Output']

filters = ['u','g','r','i','z','y']
colors={'u':'m','g':'b','r':'g','i':'y','z':'r','y':'k'}
filters=['r']

binList=[]

# Fetch the proposal ID values from the database
propids, WFDpropid, DDpropid = runInfo.fetchPropIDs(root.dbAddress['dbAddress'])

# Fetch design and strech specs from DB and scale to the length of the observing run if not 10 years
nvisitDesign, nvisitStretch, coaddedDepthDesign, coaddedDepthStretch, skyBrighntessDesign, seeingDesign = runInfo.scaleStretchDesign(root.dbAddress['dbAddress'])

# Check how many fields are requested per propID and for all proposals
# Not sure I actually need to use this anywhere...
#nFields = runInfo.fetchNFields(root.dbAddress['dbAddress'], propids)


# Plotting ranges and normalizations
mag_zpoints = coaddedDepthDesign
seeing_norm = seeingDesign
sky_zpoints = skyBrighntessDesign
nVisits_plotRange = {'all': 
                     {'u':[25, 75], 'g':[50,100], 'r':[150, 200], 'i':[150, 200], 'z':[100, 250], 'y':[100,250]},
                     'DDpropid': 
                     {'u':[6000, 10000], 'g':[2500, 5000], 'r':[5000, 8000], 'i':[5000, 8000],  'z':[7000, 10000], 'y':[5000, 8000]},
                     '216':
                     {'u':[20, 40], 'g':[20, 40], 'r':[20, 40], 'i':[20, 40], 'z':[20, 40], 'y':[20, 40]}}


# Construct a WFD SQL where clause so multiple propIDs can by WFD:
wfdWhere = ''
for i,propid in enumerate(WFDpropid):
    if i == 0:
        wfdWhere = wfdWhere+'('+'propID = %s'%propid
    else:
        wfdWhere = wfdWhere+'or propID = %s'%propid
    wfdWhere = wfdWhere+')'




# Metrics per filter 
for f in filters:
    m1 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'Nvisits'}, 
                          plotDict={'units':'Number of Visits', 
                                    'histMin':nVisits_plotRange['all'][f][0],
                                    'histMax':nVisits_plotRange['all'][f][1]})
    m2 = makeMetricConfig('CountMetric', params=['expMJD'], kwargs={'metricName':'NVisitsRatio'},
                          plotDict={'normVal':nvisitDesign[f], 'ylog':False, 'units':'Number of Visits/Designmark (%d)' %(nvisitDesign[f])})
    m3 = makeMetricConfig('MedianMetric', params=['fivesigma_modified'])
    m4 = makeMetricConfig('Coaddm5Metric',kwargs={'m5col':'fivesigma_modified'}, plotDict={'zp':float(mag_zpoints[f]), 'percentileClip':95., 'units':'Co-add (m5 - %.1f)'%mag_zpoints[f]},
                          histMerge={'histNum':6, 'legendloc':'upper right', 'color':colors[f],'label':'%s'%f} )             
    m5 = makeMetricConfig('MedianMetric', params=['perry_skybrightness'], plotDict={'zp':sky_zpoints[f], 'units':'Skybrightness - %.2f' %(sky_zpoints[f])})
    m6 = makeMetricConfig('MedianMetric', params=['finSeeing'], plotDict={'normVal':seeing_norm[f], 'units':'Median Seeing/(Expected seeing %.2f)'%(seeing_norm[f])})
    m7 = makeMetricConfig('MedianMetric', params=['airmass'], plotDict={'_units':'X'})
    m8 = makeMetricConfig('MaxMetric', params=['airmass'], plotDict={'_units':'X'})
    m9 = makeMetricConfig('MaxMetric', params=['airmass'], kwargs={'metricName':'airmass_plus_masked'}, plotDict={'plotMaskedValues':True,'_units':'X'})
    metricDict = makeDict(m1,m8,m9)
    binner = makeBinnerConfig('OpsimFieldBinner', kwargs={'badval':0}, metricDict=metricDict, constraints=["filter = \'%s\'"%f])
    binList.append(binner)




root.binners=makeDict(*binList)









