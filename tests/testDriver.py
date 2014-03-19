import numpy as np #to stop seg fault!
import unittest
import lsst.sims.operations.maf.driver as driver
from lsst.sims.operations.maf.driver.mafConfig import MafConfig
#things to make sure I exercise

#multiple binners--with kwargs, params, setupkwrd, setupParams, constraints, stackCols, plotCOnfigs, metadata

#test all the convienence functions


class TestDriver(unittest.TestCase):
    
    def setUp(self):
        self.cfgFile = 'mafconfigTest.py'
        
    def test_driver(self):
        testDriver = driver.MafDriver(configOverrideFilename=self.cfgFile)
        testDriver.run()

    def test_outputConfig(self):
        configIn = MafConfig()
        configIn.load(self.cfgFile )
        configOut = MafConfig()
        configOut.load(configIn.outputDir+'/maf_config_asRan.py')
        assert(configIn == configOut)
        
if __name__ == '__main__':
    unittest.main()
