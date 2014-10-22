import numpy as np
import numpy.ma as ma
import warnings
import unittest
import json
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.sliceMetrics as sliceMetrics


def makeDataValues(size=100, min=0., max=1., random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min())
    datavalues += min
    if random:
        randorder = np.random.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    datavalues = np.array(zip(datavalues), dtype=[('testdata', 'float')])
    return datavalues

def makeMetricData(slicer, dtype='float'):
    metricValues = np.random.rand(len(slicer)).astype(dtype)
    metricValues = ma.MaskedArray(data=metricValues,
                                  mask = np.zeros(len(slicer), 'bool'),
                                  fill_value=slicer.badval)
    return metricValues


def makeFieldData():
    """Set up sample field data."""
    # These are a subset of the fields from opsim.
    fieldId = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010]
    fieldRA = [1.4961071750760884, 4.009380232682723, 2.2738050744968632, 2.7527439701957053, 6.043715459855715,
               0.23946974745438585, 3.4768050063149119, 2.8063803008646744, 4.0630173623005916, 2.2201678117208452]
    fieldDec = [-0.25205231807872636, -0.25205228478831621, -0.25205228478831621, -0.25205228478831621, -0.25205145255075168,
                -0.25205145255075168, -0.24630904473998308, -0.24630904473998308, -0.24630894487049795, -0.24630894487049795]
    fieldId = np.array(fieldId, 'int')
    fieldRA = np.array(fieldRA, 'float')
    fieldDec = np.array(fieldDec, 'float')
    fieldData = np.core.records.fromarrays([fieldId, fieldRA, fieldDec],
                                            names = ['fieldID', 'fieldRA', 'fieldDec'])
    return fieldData

def makeOpsimDataValues(fieldData, size=10000, min=0., max=1., random=True):
    """Generate a simple array of numbers, evenly arranged between min/max, but (optional) random order."""
    datavalues = np.arange(0, size, dtype='float')
    datavalues *= (float(max) - float(min)) / (datavalues.max() - datavalues.min()) 
    datavalues += min
    if random:
        randorder = np.random.rand(size)
        randind = np.argsort(randorder)
        datavalues = datavalues[randind]
    # Add valid fieldID values to match data values
    fieldId = np.zeros(len(datavalues), 'int')
    idxs = np.random.rand(size) * len(fieldData['fieldID'])
    for i, d in enumerate(datavalues):
        fieldId[i] = fieldData[int(idxs[i])][0]
    simData = np.core.records.fromarrays([fieldId, datavalues], names=['fieldID', 'testdata'])
    return simData


class TestJSONoutUniSlicer(unittest.TestCase):
    def setUp(self):
        self.testslicer = slicers.UniSlicer()

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, 'float')
        io = self.testslicer.outputJSON(metricVal, metricName='testMetric',
                                    simDataName ='testSim', metadata='testmeta')
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['metricName'], 'testMetric')
        self.assertEqual(jsn_header['simDataName'], 'testSim')
        self.assertEqual(jsn_header['metadata'], 'testmeta')
        self.assertEqual(jsn_header['slicerName'], 'UniSlicer')
        self.assertEqual(jsn_header['slicerLen'], 1)
        self.assertEqual(len(jsn_data), 1)

class TestJSONoutOneDSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        dv = makeDataValues(1000)
        self.testslicer = slicers.OneDSlicer(sliceColName='testdata')
        self.testslicer.setupSlicer(dv)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, 'float')
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'OneDSlicer')
        self.assertEqual(jsn_header['slicerLen'], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal))
        for jsndat, binleft, mval in zip(jsn_data, self.testslicer.bins[:-1], metricVal.data):
            self.assertEqual(jsndat[0], binleft)
            self.assertEqual(jsndat[1], mval)

class TestJSONoutOneDSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        dv = makeDataValues(1000)
        self.testslicer = slicers.OneDSlicer(sliceColName='testdata')
        self.testslicer.setupSlicer(dv)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, 'float')
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'OneDSlicer')
        self.assertEqual(jsn_header['slicerLen'], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal)+1)
        for jsndat, binleft, mval in zip(jsn_data, self.testslicer.slicePoints['bins'], metricVal.data):
            self.assertEqual(jsndat[0], binleft)
            self.assertEqual(jsndat[1], mval)

class TestJSONoutHealpixSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        self.testslicer = slicers.HealpixSlicer(nside=4, verbose=False)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, 'float')
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'HealpixSlicer')
        self.assertEqual(jsn_header['slicerLen'], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal))
        for jsndat, ra, dec, mval in zip(jsn_data, self.testslicer.slicePoints['ra'],
                                         self.testslicer.slicePoints['dec'], metricVal.data):
            self.assertAlmostEqual(jsndat[0], ra/np.pi*180.)
            self.assertAlmostEqual(jsndat[1], dec/np.pi*180.)
            self.assertEqual(jsndat[2], mval)

class TestJSONoutOpsimFieldSlicer(unittest.TestCase):
    def setUp(self):
        # Set up a slicer and some metric data for that slicer.
        self.testslicer = slicers.OpsimFieldSlicer()
        self.fieldData = makeFieldData()
        self.simData = makeOpsimDataValues(self.fieldData)
        self.testslicer.setupSlicer(self.simData, self.fieldData)

    def tearDown(self):
        del self.testslicer

    def test(self):
        metricVal = makeMetricData(self.testslicer, 'float')
        io = self.testslicer.outputJSON(metricVal)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'OpsimFieldSlicer')
        self.assertEqual(jsn_header['slicerLen'], len(self.testslicer))
        self.assertEqual(len(jsn_data), len(metricVal))
        for jsndat, ra, dec, mval in zip(jsn_data, self.testslicer.slicePoints['ra'],
                                         self.testslicer.slicePoints['dec'], metricVal.data):
            self.assertAlmostEqual(jsndat[0], ra/np.pi*180.)
            self.assertAlmostEqual(jsndat[1], dec/np.pi*180.)
            self.assertEqual(jsndat[2], mval)


class TestJSONoutSliceMetric(unittest.TestCase):
    def setUp(self):
        dv = makeDataValues(1000)
        testslicer = slicers.OneDSlicer(sliceColName='testdata')
        testslicer.setupSlicer(dv)
        metricValues = makeMetricData(testslicer, 'float')
        self.testsm = sliceMetrics.BaseSliceMetric(useResultsDb=False)
        self.testsm.slicers[0] = testslicer
        self.testsm.metricValues[0] = metricValues
        self.testsm.metricNames[0] = 'testMetric'
        self.testsm.simDataNames[0] = 'testSim'
        self.testsm.metadatas[0] = 'testMeta'
        plotDict = {'units':'testUnits', 'xlabel':'myX', 'ylabel':'myY', 'title':'myTitle'}
        self.testsm.plotDicts[0] = plotDict
        testslicer2 = slicers.HealpixSlicer(nside=4, verbose=False)
        metricValues = makeMetricData(testslicer2)
        self.testsm.slicers[1] = testslicer2
        self.testsm.metricValues[1] = metricValues
        self.testsm.metricNames[1] =  'testMetric2'
        self.testsm.simDataNames[1] = 'testSim'
        self.testsm.metadatas[1] = 'testMeta'
        self.testsm.plotDicts[1] = {}

    def tearDown(self):
        del self.testsm

    def test(self):
        # Test works from slice Metric and includes plotDict.
        io = self.testsm.outputMetricJSON(0)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'OneDSlicer')
        self.assertEqual(jsn_header['xlabel'], 'myX')
        self.assertEqual(jsn_header['ylabel'], 'myY')
        self.assertEqual(jsn_header['title'], 'myTitle')
        # Redo to check default plot Dict.
        self.testsm.plotDicts[0] = {}
        io = self.testsm.outputMetricJSON(0)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        jsn_data = jsn[1]
        self.assertEqual(jsn_header['slicerName'], 'OneDSlicer')
        self.assertEqual(jsn_header['xlabel'], 'testdata ()')
        self.assertEqual(jsn_header['ylabel'], 'testMetric')
        # And check for healpix slicer.
        io = self.testsm.outputMetricJSON(1)
        jsn = json.loads(io.getvalue())
        jsn_header = jsn[0]
        self.assertEqual(jsn_header['slicerName'], 'HealpixSlicer')
        self.assertEqual(jsn_header['xlabel'], 'testMetric2')

if __name__ == '__main__':
    unittest.main()
