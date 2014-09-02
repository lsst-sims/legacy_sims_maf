# Let's try to connect to fatboy and run some metrics on stripe 82

from lsst.sims.maf.driver.mafConfig import configureSlicer, configureMetric, makeDict, configureStacker


root.dbAddress={'dbAddress':'mssql+pymssql://clue-1:wlZH2xWy@fatboy.npl.washington.edu:1433', 'dbClass':'SdssDatabase'}

root.outputDir = 'SDSSDir'
root.opsimName = 'stripe82'
root.verbose = True
root.getConfig = False
slicerList = []
nside = 512


m1 = configureMetric('MeanMetric', kwargs={'col':'psfWidth'})
m2 =  configureMetric('MaxMetric', kwargs={'col':'nStars'}, plotDict={'cbarFormat':'%i'})
m3 =  configureMetric('MaxMetric', kwargs={'col':'nGalaxy'}, plotDict={'cbarFormat':'%i'})
m4 = configureMetric('CountMetric', kwargs={'col':'psfWidth'})


metricDict = makeDict(m1,m2,m3,m4)
sqlconstraint = 'filter="r" and nStars > 0 and nGalaxy > 0'
stacker = configureStacker('SdssRADecStacker')
slicer = configureSlicer('HealpixSDSSSlicer',
                            kwargs={'nside':nside,'spatialkey1':'RA1', 'spatialkey2':'Dec1'},
                            metricDict=metricDict, stackerDict=makeDict(stacker), constraints=[sqlconstraint,])

slicerList.append(slicer)

m1 = configureMetric('MinMetric', kwargs={'col':'Dec1'})
m2 = configureMetric('MinMetric', kwargs={'col':'Dec2'})
m3 = configureMetric('MaxMetric', kwargs={'col':'Dec1'})
m4 = configureMetric('MaxMetric', kwargs={'col':'Dec2'})
metricDict = makeDict(m1,m2,m3,m4)
slicer = configureSlicer('UniSlicer',metricDict=metricDict, constraints=[sqlconstraint])
slicerList.append(slicer)


root.slicers=makeDict(*slicerList)


# OK, so p9,p10 should be a repeat of p1,p2.

#SELECT TOP 1000 [id]
#      ,[fitsFileName]
#      ,[seqFileName]
#      ,[startPos]
#      ,[length]
#      ,[time]
#      ,[season]
#      ,[filter]
#      ,[run]
#      ,[rerun]
#      ,[field]
#      ,[frame]
#      ,[ccdLoc]
#      ,[stripe]
#      ,[strip]
#      ,[eqnlScn]
#      ,[node]
#      ,[softBias]
#      ,[flux20]
#      ,[flux0]
#      ,[sky]
#      ,[camCol]
#      ,[p1]
#      ,[p2]
#      ,[p3]
#      ,[p4]
#      ,[p5]
#      ,[p6]
#      ,[p7]
#      ,[p8]
#      ,[p9]
#      ,[p10]
#      ,[bbox]
#      ,[fluxMag0]
#      ,[FluxMag0err]
#      ,[fieldid]
#      ,[nObjects]
#      ,[nGalaxy]
#      ,[nStars]
#      ,[quality]
#      ,[airmass]
#      ,[imageQualSky]
#      ,[psfWidth]
#      ,[blacklist]
#  FROM [clue].[dbo].[viewStripe82JoinAll]
#
