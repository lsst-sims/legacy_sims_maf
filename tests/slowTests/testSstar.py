import matplotlib
matplotlib.use("Agg")
import importlib
import os,sys, shutil
import unittest
from lsst.sims.maf.driver.mafConfig import MafConfig
import lsst.sims.maf.driver as driver
import glob

class TestSstar(unittest.TestCase):

    def testSstarConfig(self):
        """Load up and run the SStar config with the small example db. """
        configFile = glob.glob('../../examples/driverConfigs/sstarDriver.py')
        path, configname = os.path.split(configFile[0])
        configname = os.path.splitext(configname)[0]
        sys.path.insert(0, path)

        conf = importlib.import_module(configname)
        config = MafConfig()
        config = conf.mConfig(config, runName='opsimblitz1_1133',
                              outDir='Output', dbDir='../',slicerName='OpsimFieldSlicer')
        drive = driver.MafDriver(config)
        drive.run()

    def tearDown(self):
        if os.path.isdir('Output'):
            shutil.rmtree('Output')

if __name__ == "__main__":
    unittest.main()
