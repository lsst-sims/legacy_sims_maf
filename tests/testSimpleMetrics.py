import numpy as np #to stop seg fault!
import lsst.sims.operations.maf.metrics.simpleMetrics as sm
import lsst.sims.operations.maf.utils.testUtils as tu
testdata = tu.makeSimpleTestSet()
metrics = []
metrics.append(sm.Coaddm5Metric('m5', 'm5Metric'))
metrics.append(sm.MinMetric('seeing', 'minSeeingMetric'))
metrics.append(sm.MaxMetric('seeing', 'maxSeeingMetric'))
metrics.append(sm.MeanMetric('expmjd', 'meanExpMjdMetric'))
metrics.append(sm.RmsMetric('expmjd', 'rmsExpMjdMetric'))
metrics.append(sm.FullRangeMetric('expmjd', 'fullrangeExpMjdMetric'))
for metric in metrics:
    metric.validateData(testdata)
    print "%s -- %f"%(metric.name, metric.run(testdata))
