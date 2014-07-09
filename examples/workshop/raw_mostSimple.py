import lsst.sims.maf.driver.mafConfig
assert type(root)==lsst.sims.maf.driver.mafConfig.MafConfig, 'config is of type %s.%s instead of lsst.sims.maf.driver.mafConfig.MafConfig' % (type(root).__module__, type(root).__name__)
root.comment='runName'
root.verbose=False
root.dbAddress={'dbAddress': 'sqlite:///../../tests/opsimblitz1_1131_sqlite.db'}
root.slicers={}
root.slicers[0]=lsst.sims.maf.driver.mafConfig.SlicerConfig()
root.slicers[0].name='HealpixSlicer'
root.slicers[0].plotConfigs={}
root.slicers[0].metricDict={}
root.slicers[0].metricDict[0]=lsst.sims.maf.driver.mafConfig.MetricConfig()
root.slicers[0].metricDict[0].name='MeanMetric'
root.slicers[0].metricDict[0].params=['finSeeing']
root.slicers[0].metricDict[0].summaryStats={}
root.slicers[0].metricDict[0].summaryStats['RmsMetric']=lsst.sims.maf.driver.mafConfig.MixConfig()
root.slicers[0].constraints=['']
root.slicers[0].metadata=''
root.getConfig=True
root.outputDir='./Most_simple_out'
root.opsimName='MostSimpleExample'
