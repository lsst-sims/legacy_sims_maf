import numpy as np #to stop seg fault!
import unittest
import lsst.sims.operations.maf.driver as driver
from lsst.sims.operations.maf.driver.mafConfig import MafConfig
#things to make sure I exercise

#multiple binners--with kwargs, params, setupkwrd, setupParams, constraints, stackCols, plotCOnfigs, metadata

#test all the convienence functions


class TestDriver(unittest.TestCase):
    
    def setUp(self):
        self.cfgFiles = ['mafconfigTest.py', 'mafconfig2Test.py']
        
        
    def test_driver(self):
        for filename in self.cfgFiles:
            testDriver = driver.MafDriver(configOverrideFilename=filename)
            testDriver.run()

            configIn = MafConfig()
            configIn.load(filename)
            configOut = MafConfig()
            configOut.load(configIn.outputDir+'/maf_config_asRan.py')
            assert(configIn == configOut)

       
if __name__ == '__main__':
    unittest.main()
