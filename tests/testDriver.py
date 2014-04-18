import numpy as np 
import matplotlib
matplotlib.use('Agg')
import unittest
import lsst.sims.maf.driver as driver
from lsst.sims.maf.driver.mafConfig import MafConfig
import glob
from subprocess import Popen
import os
import inspect
import shutil


#things to make sure I exercise

#multiple binners--with kwargs, params, setupkwrd, setupParams, constraints, stackCols, plotCOnfigs, metadata

# Need to test all the convienence functions


class TestDriver(unittest.TestCase):
    
    def setUp(self):
        self.cfgFiles = ['mafconfigTest.cfg', 'mafconfig2Test.cfg']
        self.filepath = os.environ['SIMS_MAF_DIR']+'/tests/'
        #self.filepath=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+'/'
    def test_overwrite(self):
        filename='mafconfigTest3.cfg'
        configIn = MafConfig()
        configIn.load(self.filepath+filename)
        self.assertRaises(Exception, driver.MafDriver,**{'configOverrideFilename':'filename'})
    
    def test_driver(self):
        for filename in self.cfgFiles:
            configIn = MafConfig()
            configIn.load(self.filepath+filename)
            nnpz = glob.glob(configIn.outputDir+'/*.npz')
            if len(nnpz) > 0:
                ack = Popen('rm '+configIn.outputDir+'/*.npz', shell=True).wait()
            
            testDriver = driver.MafDriver(configOverrideFilename=self.filepath+filename)
            testDriver.run()

            configOut = MafConfig()
            configOut.load(configIn.outputDir+'/maf_config_asRan.py')
            assert(configIn == configOut)
            nout=0
            for i,binner in enumerate(configIn.binners):
                if configIn.binners[i].name != 'HourglassBinner':
                    nout += len(configIn.binners[i].constraints)*len(configIn.binners[i].metricDict)
            nnpz = glob.glob(configIn.outputDir+'/*.npz')
            assert(nout == len(nnpz))
            
    def tearDown(self):
        if os.path.isdir('Output'):
            shutil.rmtree('Output')
        if os.path.isdir('Output2'):
            shutil.rmtree('Output2')
       
if __name__ == '__main__':
    unittest.main()
