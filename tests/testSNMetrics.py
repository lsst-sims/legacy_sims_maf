from builtins import zip
import matplotlib
matplotlib.use("Agg")
import numpy as np
import unittest
#import lsst.sims.maf.metrics as metrics
import lsst.utils.tests
from lsst.sims.maf.utils.snUtils import Lims, Reference_Data
from lsst.sims.maf.metrics.snCadenceMetric import SNcadenceMetric
from lsst.sims.maf.metrics.snSNRMetric import SNSNRMetric

m5_ref = dict(
    zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


class TestSNmetrics(unittest.TestCase):

    def testSNCadenceMetric(self):
        """Test the SN cadence metric """

        # Load required SN info to run the metric
        band = 'r'
        SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
        mag_range = [21., 25.5]  # WFD mag range
        dt_range = [0.5, 30.]  # WFD dt range
        Li_files = ['Li_SNCosmo_-2.0_0.2.npy']
        mag_to_flux_files = ['Mag_to_Flux_SNCosmo.npy']
        lim_sn = Lims(Li_files, mag_to_flux_files, band, SNR[band],
                      mag_range=mag_range, dt_range=dt_range)

        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime']
        types = ['f8']*len(names)
        names += ['night']
        types += ['i2']
        names += ['filter']
        types += ['O']

        day0 = 59000
        daylast = day0+250
        cadence = 5
        dayobs = np.arange(day0, daylast, cadence)
        npts = len(dayobs)
        data = np.zeros(npts, dtype=list(zip(names, types)))
        data['observationStartMJD'] = dayobs
        data['night'] = np.floor(data['observationStartMJD']-day0)
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 15.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['filter'] = band

        # Run the metric with these fake data
        slicePoint = [0]
        metric = SNcadenceMetric(lim_sn=lim_sn, coadd=False)
        result = metric.run(data, slicePoint)

        # And the result should be...
        result_ref = 0.3743514

        assert(np.abs(result-result_ref) < 1.e-5)

    def testSNSNRMetric(self):
        """Test the SN SNR metric """

        # Load required SN info to run the metric
        band = 'r'
        z = 0.3
        season = 1.
        Li_files = ['Li_SNCosmo_-2.0_0.2.npy']
        mag_to_flux_files = ['Mag_to_Flux_SNCosmo.npy']
        names_ref = ['SNCosmo']
        coadd = False

        lim_sn = Reference_Data(Li_files, mag_to_flux_files, band, z)

        # Define fake data
        names = ['observationStartMJD', 'fieldRA', 'fieldDec',
                 'fiveSigmaDepth', 'visitExposureTime', 'numExposures', 'visitTime', 'season']
        types = ['f8']*len(names)
        names += ['night']
        types += ['i2']
        names += ['filter']
        types += ['O']

        dayobs = [59948.31957176, 59959.2821412, 59970.26134259,
                  59973.25978009, 59976.26383102, 59988.20670139, 59991.18412037,
                  60004.1853588, 60032.08975694, 60045.11981481, 60047.98747685,
                  60060.02083333, 60071.986875, 60075.96452546]
        day0 = np.min(dayobs)
        npts = len(dayobs)
        data = np.zeros(npts, dtype=list(zip(names, types)))

        data['observationStartMJD'] = dayobs
        data['night'] = np.floor(data['observationStartMJD']-day0)
        data['fiveSigmaDepth'] = m5_ref[band]
        data['visitExposureTime'] = 15.
        data['numExposures'] = 2
        data['visitTime'] = 2.*15.
        data['season'] = season
        data['filter'] = band

        # Run the metric with these fake data
        slicePoint = [0]
        metric = SNSNRMetric(
            lim_sn=lim_sn, coadd=coadd, names_ref=names_ref, season=season, z=z,
            config_fake='Fake_cadence.yaml')

        result = metric.run(data, slicePoint)

        # And the result should be...
        result_ref = 0.5234375

        assert(np.abs(result-result_ref) < 1.e-5)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
