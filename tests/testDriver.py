import numpy as np #to stop seg fault!
import unittest
import lsst.sims.operations.maf.driver as driver
from lsst.sims.operations.maf.driver.mafConfig import MafConfig
import glob
from subprocess import Popen

#things to make sure I exercise

#multiple binners--with kwargs, params, setupkwrd, setupParams, constraints, stackCols, plotCOnfigs, metadata

#test all the convienence functions


class TestDriver(unittest.TestCase):
    
    def setUp(self):
        self.cfgFiles = ['mafconfigTest.py', 'mafconfig2Test.py']
        
    def test_overwrite(self):
        filename='mafconfigTest3.cfg'
        configIn = MafConfig()
        configIn.load(filename)
        self.assertRaises(Exception, driver.MafDriver,**{'configOverrideFilename':'filename'})
    
    def test_driver(self):
        for filename in self.cfgFiles:
            configIn = MafConfig()
            configIn.load(filename)
            nnpz = glob.glob(configIn.outputDir+'/*.npz')
            if len(nnpz) > 0:
                ack = Popen('rm '+configIn.outputDir+'/*.npz', shell=True).wait()
            
            testDriver = driver.MafDriver(configOverrideFilename=filename)
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
            

       
if __name__ == '__main__':
    unittest.main()
