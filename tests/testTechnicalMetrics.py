import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics

class TestTechnicalMetrics(unittest.TestCase):

    def testOpenShutterFractionMetric(self):
        """
        Test the open shutter fraction metric.
        """
        nvisit = 10
        exptime = 30.
        slewtime = 30.
        visitExpTime = np.ones(nvisit, dtype='float')*exptime
        slewTime = np.ones(nvisit, dtype='float')*slewtime
        data = np.core.records.fromarrays([visitExpTime, slewTime], names=['visitExpTime', 'slewTime'])
        metric = metrics.OpenShutterFractionMetric(readTime=0, shutterTime=0)
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
