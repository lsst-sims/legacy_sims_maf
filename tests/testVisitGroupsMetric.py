import numpy as np
import unittest
import lsst.sims.maf.metrics as metrics


class TestVisitGroupsMetric(unittest.TestCase):        
    def testVisitGroups(self):
        """Test visit groups (solar system groups) metric."""
        # Set up some simple test data.
        tmin = 15.0/60./24.0
        tmax = 90./60./24.0
        tstart = 49406.00
        # The test data needs to have times for all observations: set up the times in two stages .. the night 
        night = np.array([0, 0, 0, 0, 0, 0, 
                          1, 1, 
                          2, 
                          3, 3, 
                          31, 31, 
                          32, 32, 
                          33, 33, 
                          34, 34, 34, 34,
                          35, 35, 35, 
                          36, 36, 36, 
                          37, 37, 37, 
                          38, 38, 38], 'int')
        #   .. and the time within the night.
        expmjd = np.array([tstart, tstart+tmin/10.0, tstart+tmin+tmin/2.0, tstart+tmin*2+tmax,
                           tstart+2*tmax, tstart+2*tmax+tmin/2.0, #n0
                           tstart, tstart+tmax, # n1 .. only 2 obs but should make a pair
                           tstart, # n2 - only 1 obs, should not make a pair
                           tstart, tstart+tmin, # n3 .. only 2 obs but should make a pair
                           tstart, tstart+tmin, tstart, tstart+tmin,
                           tstart, tstart+tmin, #n31/32/33 - a pair on each night
                           tstart, tstart+tmin/10.0, tstart+tmax*2,
                           tstart+tmax*2+tmin/10.0, # n34 .. should make no pairs
                           tstart, tstart+tmin/10.0, tstart+tmax, # n35 should make 2.5 (too close at start)
                           tstart, tstart+tmax, tstart+tmax+tmin/10.0, #n36 should make 2.5 pairs (too close at end)
                           tstart, tstart+tmin, tstart+tmax, # n37 - 3 (regular 3)
                           tstart, tstart+tmax, tstart+tmax*2 # n38 - 3 (three, but long spacing)
                           ], 'float')
        expmjd = expmjd + night
        testdata = np.core.records.fromarrays([expmjd, night], names=['expmjd', 'night'])
        # Set up metric.
        testmetric = metrics.VisitGroupsMetric(timesCol='expmjd', nightsCol='night',
                                                deltaTmin=tmin, deltaTmax=tmax, minNVisits=2,
                                                window=5, minNNights=3)
        # and set up a copy, with a higher number of min visits per night 
        testmetric2 = metrics.VisitGroupsMetric(timesCol='expmjd', nightsCol='night', deltaTmin=tmin, deltaTmax=tmax, 
                                                minNVisits=3, window=5, minNNights=3)
        # Run metric for expected results.
        metricval = testmetric.run(testdata)
        # These are the expected results, based on the times above.
        expected_nights = np.array([0, 1, 3, 31, 32, 33, 35, 36, 37, 38])
        expected_numvisits = np.array([5.0, 2, 2, 2, 2, 2, 2.5, 2.5, 3, 3])
        np.testing.assert_equal(metricval['visits'], expected_numvisits)
        np.testing.assert_equal(metricval['nights'], expected_nights)
        # Test reduce methods.
        self.assertEqual(testmetric.reduceMedian(metricval), np.median(expected_numvisits))
        self.assertEqual(testmetric.reduceNNightsWithNVisits(metricval), len(expected_nights))
        self.assertEqual(testmetric2.reduceNNightsWithNVisits(metricval), 3)
        self.assertEqual(testmetric.reduceNVisitsInWindow(metricval), 11)
        self.assertEqual(testmetric2.reduceNNightsInWindow(metricval), 2)
        self.assertEqual(testmetric.reduceNLunations(metricval), 2)
        # Test with a longer (but simpler) date range.
        indnight = np.array([0, 1, 2, 3, 4, 5, 31, 32, 33, 34, 61, 62, 63, 121, 122, 123], 'int')
        indtimes = np.array([tstart, tstart+tmin, tstart+tmax], 'float')
        expmjd = []
        night = []
        for n in indnight:
            for t in indtimes:            
                expmjd.append(float(n + t))
                night.append(n)
        expmjd = np.array(expmjd)
        night = np.array(night)
        testdata = np.core.records.fromarrays([expmjd, night], names=['expmjd', 'night'])
        metricval = testmetric.run(testdata)
        self.assertEqual(testmetric.reduceNLunations(metricval), 4)
        self.assertEqual(testmetric.reduceMaxSeqLunations(metricval), 3)

                
if __name__ == '__main__':

    unittest.main()
