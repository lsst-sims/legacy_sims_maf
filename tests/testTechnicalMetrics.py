import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics

class TestTechnicalMetrics(unittest.TestCase):

    def testNChangesMetric(self):
        """
        Test the NChanges metric.
        """
        filters = np.array(['u', 'u', 'g', 'g', 'r'])
        visitTimes = np.arange(0, filters.size, 1)
        data = np.core.records.fromarrays([visitTimes, filters],
                                          names=['expMJD', 'filter'])
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 2)
        filters = np.array(['u', 'g', 'u', 'g', 'r'])
        data = np.core.records.fromarrays([visitTimes, filters],
                                          names=['expMJD', 'filter'])
        metric = metrics.NChangesMetric()
        result = metric.run(data)
        self.assertEqual(result, 4)

    def testDeltaTimeChangesMetric(self):
        """
        Test the DeltaTime metric.
        """
        filters = np.array(['u', 'g'])
        visitTimes = np.random.rand(2) * 10.0 #days
        data = np.core.records.fromarrays([visitTimes, filters],
                                          names=['expMJD', 'filter'])
        metric = metrics.DeltaTimeChangesMetric()
        result = metric.run(data) # minutes
        self.assertEqual(result, np.diff(visitTimes)*24.0*60.0)
        filters = np.array(['u', 'u', 'g', 'g', 'r'])
        visitTimes = np.random.rand(filters.size) * 10.0
        visitTimes.sort()
        data = np.core.records.fromarrays([visitTimes, filters],
                                          names=['expMJD', 'filter'])
        result = metric.run(data)
        self.assertEqual(result.size, 2)

    def testTeffMetric(self):
        """
        Test the Teff (time_effective) metric.
        """
        filters = np.array(['g', 'g', 'g', 'g', 'g'])
        m5 = np.zeros(len(filters), float) + 25.0
        data = np.core.records.fromarrays([m5, filters],
                                        names=['fiveSigmaDepth', 'filter'])
        metric = metrics.TeffMetric(fiducialDepth={'g':25}, teffBase=30.0)
        result = metric.run(data)
        self.assertEqual(result, 30.0*m5.size)

    def testOpenShutterFractionMetric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.
        slewtime = 30.
        visitExpTime = np.ones(nvisit, dtype='float')*exptime
        visitTime = np.ones(nvisit, dtype='float')*(exptime+0.0)
        slewTime = np.ones(nvisit, dtype='float')*slewtime
        data = np.core.records.fromarrays([visitExpTime, visitTime, slewTime],
                                          names=['visitExpTime', 'visitTime', 'slewTime'])
        metric = metrics.OpenShutterFractionMetric()
        result = metric.run(data)
        self.assertEqual(result, .5)

    def testCompletenessMetric(self):
        """
        Test the completeness metric.
        """
        # Generate some test data.
        data = np.zeros(600, dtype=zip(['filter'],['|S1']))
        data['filter'][:100] = 'u'
        data['filter'][100:200] = 'g'
        data['filter'][200:300]= 'r'
        data['filter'][300:400] = 'i'
        data['filter'][400:550] = 'z'
        data['filter'][550:600] = 'y'
        slicePoint = [0]
        # Test completeness metric when requesting all filters.
        metric = metrics.CompletenessMetric(u=100, g=100, r=100, i=100, z=100, y=100)
        completeness = metric.run(data, slicePoint)
        assert(metric.reduceu(completeness) == 1)
        assert(metric.reduceg(completeness) == 1)
        assert(metric.reducer(completeness) == 1)
        assert(metric.reducei(completeness) == 1)
        assert(metric.reducez(completeness) == 1.5)
        assert(metric.reducey(completeness) == 0.5)
        assert(metric.reduceJoint(completeness) == 0.5)
        # Test completeness metric when requesting only some filters.
        metric = metrics.CompletenessMetric(u=0, g=100, r=100, i=100, z=100, y=100)
        completeness = metric.run(data, slicePoint)
        assert(metric.reduceu(completeness) == 1)
        assert(metric.reduceg(completeness) == 1)
        assert(metric.reducer(completeness) == 1)
        assert(metric.reducei(completeness) == 1)
        assert(metric.reducez(completeness) == 1.5)
        assert(metric.reducey(completeness) == 0.5)
        assert(metric.reduceJoint(completeness) == 0.5)
        # Test completeness metric when some filters not observed at all.
        metric = metrics.CompletenessMetric(u=100, g=100, r=100, i=100, z=100, y=100)
        data['filter'][550:600] = 'z'
        data['filter'][:100] = 'g'
        completeness = metric.run(data, slicePoint)
        assert(metric.reduceu(completeness) == 0)
        assert(metric.reduceg(completeness) == 2)
        assert(metric.reducer(completeness) == 1)
        assert(metric.reducei(completeness) == 1)
        assert(metric.reducez(completeness) == 2)
        assert(metric.reducey(completeness) == 0)
        assert(metric.reduceJoint(completeness) == 0)
        # And test that if you forget to set any requested visits, that you get the useful error message
        self.assertRaises(ValueError, metrics.CompletenessMetric, 'filter')


if __name__ == '__main__':
    unittest.main()
