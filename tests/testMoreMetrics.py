import numpy as np
import healpy as hp
import unittest
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.utils as utils

class TestMoreMetrics(unittest.TestCase):

    def testCompletenessMetric(self):
        """Test the completeness metric."""
        # Generate some test data.
        data = np.zeros(600, dtype=zip(['filter'],['|S1']))
        data['filter'][:100] = 'u'
        data['filter'][100:200] = 'g'
        data['filter'][200:300]= 'r'
        data['filter'][300:400] = 'i'
        data['filter'][400:550] = 'z'
        data['filter'][550:600] = 'y'
        # Test completeness metric when requesting all filters.
        metric = metrics.CompletenessMetric(u=100, g=100, r=100, i=100, z=100, y=100)
        completeness = metric.run(data)
        assert(metric.reduceu(completeness) == 1)
        assert(metric.reduceg(completeness) == 1)
        assert(metric.reducer(completeness) == 1)
        assert(metric.reducei(completeness) == 1)
        assert(metric.reducez(completeness) == 1.5)
        assert(metric.reducey(completeness) == 0.5)
        assert(metric.reduceJoint(completeness) == 0.5)
        # Test completeness metric when requesting only some filters. 
        metric = metrics.CompletenessMetric(u=0, g=100, r=100, i=100, z=100, y=100)
        completeness = metric.run(data)
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
        completeness = metric.run(data)
        assert(metric.reduceu(completeness) == 0)
        assert(metric.reduceg(completeness) == 2)
        assert(metric.reducer(completeness) == 1)
        assert(metric.reducei(completeness) == 1)
        assert(metric.reducez(completeness) == 2)
        assert(metric.reducey(completeness) == 0)
        assert(metric.reduceJoint(completeness) == 0)
        # And test that if you forget to set any requested visits, that you get the useful error message
        self.assertRaises(ValueError, metrics.CompletenessMetric, 'filter')        
    '''
    def testHourglassMetric(self):
        """Test the hourglass metric """
        names = [ 'expMJD', 'night','filter']
        types = [float,float,str]
        data = np.zeros(10, dtype = zip(names,types))
        data['night'] = np.round(np.arange(0,2,.1))[:10]
        data['expMJD'] = np.sort(np.random.rand(10))+data['night'] 
        data['filter'] = 'r'

        metric = metrics.HourglassMetric()
        result = metric.run(data)
        pernight = result['pernight']
        perfilter = result['perfilter']
        # Check that the format is right at least
        assert(np.size(perfilter) == 2*data.size)
        assert(len(pernight.dtype.names) == 9)
    
    def testinDevelopmentMetrics(self):
        """ Test Metrics in Development, just passes and ignores"""
        pass
    '''
    def testRadiusObsMetric(self):
        """Test the RadiusObsMetric """
        ra = 0.
        dec = 0.
        names=['fieldRA','fieldDec']
        dt = ['float']*2
        
        data = np.zeros(3, dtype=zip(names,dt))
        data['fieldDec'] = [-.1,0,.1]

        metric = metrics.RadiusObsMetric()
        result = metric.run(data,0.,0.)
        for i,r in enumerate(result):
            np.testing.assert_almost_equal(r, abs(data['fieldDec'][i]))
        assert(metric.reduceMean(result) == np.mean(result))
        assert(metric.reduceRMS(result) == np.std(result))
        np.testing.assert_almost_equal(metric.reduceFullRange(result),
               np.max(np.abs(data['fieldDec']))-np.min(np.abs(data['fieldDec'])))
        
 
    def testParallaxMetric(self):
        """Test the parallax metric """
        
        names = ['expMJD','finSeeing', 'fivesigma_modified', 'fieldRA', 'fieldDec', 'filter']
        types = [float, float,float,float,float,'|S1']
        data = np.zeros(700, dtype=zip(names,types))
        normFlags = [False, True]
        for flag in normFlags:
            data['expMJD'] = np.arange(700)+56762
            data['finSeeing'] = 0.7
            data['filter'][0:100] = 'r'
            data['filter'][100:200] = 'u'
            data['filter'][200:] = 'g'
            data['fivesigma_modified'] = 24.
            stacker = utils.ParallaxFactor()
            data = stacker.run(data)
            baseline = metrics.ParallaxMetric(normalize=flag).run(data)
            data['finSeeing'] = data['finSeeing']+.3
            worse1 = metrics.ParallaxMetric(normalize=flag).run(data)
            worse2 = metrics.ParallaxMetric(normalize=flag,rmag=22.).run(data)
            worse3 = metrics.ParallaxMetric(normalize=flag,rmag=22.).run(data[0:300])
            data['fivesigma_modified'] = data['fivesigma_modified']-1.
            worse4 = metrics.ParallaxMetric(normalize=flag,rmag=22.).run(data[0:300])
            # Make sure the RMS increases as seeing increases, the star gets fainter, the background gets brighter, or the baseline decreases.
            if flag:
                pass
                # hmm, I need to think how to test the scheduling
                #assert(worse1 < baseline)
                #assert(worse2 < worse1)
                #assert(worse3 < worse2) 
                #assert(worse4 < worse3)
            else:
                assert(worse1 > baseline)
                assert(worse2 > worse1)
                assert(worse3 > worse2)
                assert(worse4 > worse3)

    def testProperMotionMetric(self):
        """Test the ProperMotion metric """
        names = ['expMJD','finSeeing', 'fivesigma_modified', 'fieldRA', 'fieldDec', 'filter']
        types = [float, float,float,float,float,'|S1']
        data = np.zeros(700, dtype=zip(names,types))
        normFlags = [False, True]
        for flag in normFlags:
            data['expMJD'] = np.arange(700)+56762
            data['finSeeing'] = 0.7
            data['filter'][0:100] = 'r'
            data['filter'][100:200] = 'u'
            data['filter'][200:] = 'g'
            data['fivesigma_modified'] = 24.
            stacker = utils.ParallaxFactor()
            data = stacker.run(data)
            baseline = metrics.ProperMotionMetric(normalize=flag).run(data)
            data['finSeeing'] = data['finSeeing']+.3
            worse1 = metrics.ProperMotionMetric(normalize=flag).run(data)
            worse2 = metrics.ProperMotionMetric(normalize=flag,rmag=22.).run(data)
            worse3 = metrics.ProperMotionMetric(normalize=flag,rmag=22.).run(data[0:300])
            data['fivesigma_modified'] = data['fivesigma_modified']-1.
            worse4 = metrics.ProperMotionMetric(normalize=flag, rmag=22.).run(data[0:300])
            # Make sure the RMS increases as seeing increases, the star gets fainter, the background gets brighter, or the baseline decreases.
            if flag:
                #assert(worse1 < baseline)
                #assert(worse2 < worse1) # When normalized, 'perfect' survey assumed to have same seeing and limiting mags.
                assert(worse3 < worse2)
                assert(worse4 < worse3)
            else:
                assert(worse1 > baseline)
                assert(worse2 > worse1)
                assert(worse3 > worse2)
                assert(worse4 > worse3)

    def testSNMetric(self):
        """Test the SN Cadence Metric """
        names = ['expMJD', 'filter', 'fivesigma_modified']
        types = [float,'|S1', float]
        data = np.zeros(700, dtype=zip(names,types))
        data['expMJD'] = np.arange(0.,100.,1/7.) # So, 100 days are well sampled in 2 filters
        data['filter']= 'r'
        data['filter'][np.arange(0,700,2)] = 'g'
        data['fivesigma_modified'] = 30.
        metric = metrics.SupernovaMetric()
        result = metric.run(data)
        np.testing.assert_array_almost_equal(metric.reduceMedianMaxGap(result),  1/7.)
        assert(metric.reduceNsequences(result) == 10)
        assert((metric.reduceMedianNobs(result) <  561) & (metric.reduceMedianNobs(result) >  385) )

    def testTemplateExists(self):
        """Test the TemplateExistsMetric """
        names = ['finSeeing', 'expMJD']
        types=[float,float]
        data = np.zeros(10,dtype=zip(names,types))
        data['finSeeing'] = [2.,2.,3.,1.,1.,1.,0.5,1.,0.4,1.]
        data['expMJD'] = np.arange(10)
        # so here we have 4 images w/o good previous templates
        metric = metrics.TemplateExistsMetric()
        result = metric.run(data)
        assert(result == 6./10.)

    def testf0Nv(self):
        """Test the f0Nv metric """
        nside=128
        metric = metrics.f0Nv('ack',nside=nside, Nvisit=825,Asky=18000.)
        npix = hp.nside2npix(nside)
        names=['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names,types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        result1 = metric.run(data)
        deginsph = 129600./np.pi
        np.testing.assert_almost_equal(result1*18000., deginsph)
        data['blah'][:data.size/2]=0
        result2 =  metric.run(data)
        np.testing.assert_almost_equal(result2*18000., deginsph/2.)

    def testf0Area(self):
        """Test f0Area metric """
        nside=128
        metric = metrics.f0Area('ack',nside=nside, Nvisit=825,Asky=18000.)
        npix = hp.nside2npix(nside)
        names=['blah']
        types = [float]
        data = np.zeros(npix, dtype=zip(names,types))
        # Set all the pixels to have 826 counts
        data['blah'] = data['blah']+826
        result1 = metric.run(data)
        
        np.testing.assert_almost_equal(result1*825, 826)
        data['blah'][:data.size/2]=0
        result2 =  metric.run(data)
        

    def testUniformityMetric(self):
        names = ['expMJD']
        types=[float]
        data = np.zeros(100, dtype=zip(names,types))
        metric = metrics.UniformityMetric(dayStart=0.)
        result1 = metric.run(data)
        # If all the observations are on the 1st day, should be 1
        assert(result1 == 1)
        data['expMJD'] = data['expMJD']+365.25*10
        result2 = metric.run(data)
        # All on last day should also be 1
        assert(result1 == 1)
        # Make a perfectly uniform dist
        data['expMJD'] = np.arange(0.,365.25*10,365.25*10/100)
        result3 = metric.run(data)
        # Result should be zero for uniform
        np.testing.assert_almost_equal(result3, 0.)
        # A single obseravtion should give a result of 1
        data = np.zeros(1, dtype=zip(names,types))
        result4 = metric.run(data)
        assert(result4 == 1)
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoreMetrics)
    unittest.TextTestRunner(verbosity=2).run(suite)

