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


class TestDriver(unittest.TestCase):
    
    def setUp(self):
        # Files to loop over
        self.cfgFiles = ['mafconfigTest.cfg']
        # Files that get created by those configs
        self.outputFiles= [['OpsimTest_CoaddedM5__g_and_night_lt_15_HEAL.npz','OpsimTest_CoaddedM5__g_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15_OPSI.npz','OpsimTest_CoaddedM5__g_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15dith_HEAL.npz','OpsimTest_CoaddedM5__g_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_CoaddedM5__g_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15_HEAL.npz','OpsimTest_CoaddedM5__r_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15_OPSI.npz','OpsimTest_CoaddedM5__r_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15dith_HEAL.npz','OpsimTest_CoaddedM5__r_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_CoaddedM5__r_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15_HEAL.npz','OpsimTest_Count_expMJD__g_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15_OPSI.npz','OpsimTest_Count_expMJD__g_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15dith_HEAL.npz','OpsimTest_Count_expMJD__g_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_Count_expMJD__g_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15_HEAL.npz','OpsimTest_Count_expMJD__r_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15_OPSI.npz','OpsimTest_Count_expMJD__r_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15dith_HEAL.npz','OpsimTest_Count_expMJD__r_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_Count_expMJD__r_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_Count_normairmass__g_and_night_lt_15_ONED.npz','OpsimTest_Count_normairmass__g_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_Count_normairmass__r_and_night_lt_15_ONED.npz','OpsimTest_Count_normairmass__r_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_Count_slewDist__g_and_night_lt_15_ONED.npz','OpsimTest_Count_slewDist__g_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_Count_slewDist__r_and_night_lt_15_ONED.npz','OpsimTest_Count_slewDist__r_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_Mean_airmass__r_and_night_lt_15_UNIS.npz','OpsimTest_Mean_normairmass__g_and_night_lt_15_OPSI.npz','OpsimTest_Mean_normairmass__g_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Mean_normairmass__g_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_Mean_normairmass__r_and_night_lt_15_OPSI.npz','OpsimTest_Mean_normairmass__r_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Mean_normairmass__r_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_Min_airmass__g_and_night_lt_15_OPSI.npz','OpsimTest_Min_airmass__g_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Min_airmass__g_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest_Min_airmass__r_and_night_lt_15_OPSI.npz','OpsimTest_Min_airmass__r_and_night_lt_15_OPSI_Histogram.pdf','OpsimTest_Min_airmass__r_and_night_lt_15_OPSI_SkyMap.pdf','OpsimTest__OpsimTest_Min_airmass_OPSI_hist.pdf','OpsimTest_hourglass__r_and_night_lt_15_HOUR_hr.pdf','OpsimTest_normAir_hex__g_and_night_lt_15_ONED.npz','OpsimTest_normAir_hex__g_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_normAir_hex__r_and_night_lt_15_ONED.npz','OpsimTest_normAir_hex__r_and_night_lt_15_ONED_BinnedData.pdf','OpsimTest_parallax__g_and_night_lt_15_HEAL.npz','OpsimTest_parallax__g_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_parallax__g_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_parallax__g_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_parallax__g_and_night_lt_15dith_HEAL.npz','OpsimTest_parallax__g_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_parallax__g_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_parallax__g_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_parallax__r_and_night_lt_15_HEAL.npz','OpsimTest_parallax__r_and_night_lt_15_HEAL_Histogram.pdf','OpsimTest_parallax__r_and_night_lt_15_HEAL_PowerSpectrum.pdf','OpsimTest_parallax__r_and_night_lt_15_HEAL_SkyMap.pdf','OpsimTest_parallax__r_and_night_lt_15dith_HEAL.npz','OpsimTest_parallax__r_and_night_lt_15dith_HEAL_Histogram.pdf','OpsimTest_parallax__r_and_night_lt_15dith_HEAL_PowerSpectrum.pdf','OpsimTest_parallax__r_and_night_lt_15dith_HEAL_SkyMap.pdf','OpsimTest_Count_fivesigma_modified__g_ONED.npz','OpsimTest_Count_fivesigma_modified__g_ONED_BinnedData.pdf','OpsimTest_Count_fivesigma_modified__r_ONED.npz','OpsimTest_Count_fivesigma_modified__r_ONED_BinnedData.pdf','OpsimTest__OpsimTest_Count_fivesigma_modified_ONED_hist.pdf','ResultsSummary.dat','configDetails.txt','configSummary.txt','date_version_ran.dat','maf_config_asRan.py','outputFiles.npy','summaryStats.dat']]
        self.filepath = os.environ['SIMS_MAF_DIR']+'/tests/'


    def test_overwrite(self):
        """Make sure driver throws an error if files would get clobbered or SQL constraints are not unique."""
        filename = 'mafconfigOverwrite.cfg'
        configIn = MafConfig()
        configIn.load(self.filepath+filename)
        self.assertRaises(Exception, driver.MafDriver,**{'configvalues':configIn})
        filename = 'mafconfigSQLError.cfg'
        configIn = MafConfig()
        configIn.load(self.filepath+filename)
        self.assertRaises(Exception, driver.MafDriver,**{'configvalues':configIn})

    def test_png(self):
        """Test that a config that specifies png files makes pngs """
        configIn = MafConfig()
        configIn.load(self.filepath+'mafconfigpng.cfg')
        expectFiles=['OpsimTest_Count_expMJD__r_HEAL_Histogram.png',
                     'OpsimTest_Count_expMJD__r_HEAL_PowerSpectrum.png',
                     'OpsimTest_Count_expMJD__r_HEAL_SkyMap.png']
        testDriver = driver.MafDriver(configIn)
        testDriver.run()
        for filename in expectFiles:
            assert(os.path.isfile(configIn.outputDir+'/'+filename))

        
    def test_driver(self):
        """Use a large config file to exercise all aspects of the driver. """    
        for filename, outfiles in zip(self.cfgFiles, self.outputFiles):
            configIn = MafConfig()
            configIn.load(self.filepath+filename)
            nnpz = glob.glob(configIn.outputDir+'/*.npz')
            if len(nnpz) > 0:
                ack = Popen('rm '+configIn.outputDir+'/*.npz', shell=True).wait()


            testDriver = driver.MafDriver(configIn)
            testDriver.run()

            configOut = MafConfig()
            configOut.load(configIn.outputDir+'/maf_config_asRan.py')
            assert(configIn == configOut)
            nout=0
            for j,slicer in enumerate(configIn.slicers):
                if configIn.slicers[j].name != 'HourglassSlicer':
                    nout += len(configIn.slicers[j].constraints)*len(configIn.slicers[j].metricDict)

            nnpz = glob.glob(configIn.outputDir+'/*.npz')
            assert(os.path.isfile(configIn.outputDir+'/date_version_ran.dat'))
            assert(os.path.isfile(configIn.outputDir+'/summaryStats.dat'))
            for filename in outfiles:
                if not os.path.isfile(configIn.outputDir+'/'+filename):
                    print 'missing file %s'%filename
                assert(os.path.isfile(configIn.outputDir+'/'+filename))
            assert(nout == len(nnpz))

            #check that there's something in the summary stats file:
            names = ['opsimname','slicer_name','sql_where', 'metric_name','summary_stat_name','value']
            types = ['|S25','|S25','|S25','|S25','|S25', float]
            ss = np.genfromtxt('Output/summaryStats.dat', delimiter=',', comments='#', dtype=zip(names,types) )
            for value in ss['value']:
                assert(value > 0.)

    def tearDown(self):
        if os.path.isdir('Output'):
            shutil.rmtree('Output')
       
if __name__ == '__main__':
    unittest.main()



