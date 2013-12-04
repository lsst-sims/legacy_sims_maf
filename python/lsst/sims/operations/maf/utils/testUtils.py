import numpy
import numpy.random as nrand

def makeSimpleTestSet():
    SEED = 1
    nrand.seed(SEED)
    cols = ['m5', 'seeing', 'expmjd', 'filter']
    m5arr = nrand.random(1000)*2.5 + 20.
    seeing = nrand.lognormal(mean=-0.22, sigma=0.2, size=1000.)
    expmjd = nrand.random(1000)*10.0 + 5200.
    expmjd = numpy.sort(expmjd)
    night = numpy.floor(expmjd-0.16)
    filterMap = dict([(i, el) for i, el in enumerate('ugrizy')])
    ra = nrand.rand(1000)*numpy.radians(10) + numpy.radians(20.0)
    dec = nrand.rand(1000)*numpy.radians(10) + numpy.radians(20.0)
    f = [filterMap[i] for i in numpy.floor(nrand.random(1000)*10.%6)]
    return numpy.array(zip(m5arr, seeing, expmjd, night, f, ra, dec), 
                       dtype=[('m5', float), 
                              ('seeing', float),
                              ('expMJD', float),
                              ('night', int),
                              ('filter', (str, 1)),
                              ('fieldra', float),
                              ('fielddec', float)])


