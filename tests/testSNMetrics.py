from builtins import zip
import numpy as np
import unittest
import lsst.utils.tests
from lsst.sims.maf.utils.snUtils import Lims, ReferenceData
from lsst.sims.maf.utils.snNSNUtils import Throughputs, Telescope
from lsst.sims.maf.metrics import SNCadenceMetric
from lsst.sims.maf.metrics import SNSNRMetric
from lsst.sims.maf.metrics import SNSLMetric
from lsst.sims.maf.metrics import SNNSNMetric

import os
import warnings
import healpy as hp
import time

m5_ref = dict(
    zip('ugrizy', [23.60, 24.83, 24.38, 23.92, 23.35, 22.44]))


def Observations_band(day0=59000, daymin=59000, cadence=3., season_length=140., band='r'):
    # Define fake data
    names = ['observationStartMJD', 'fieldRA', 'fieldDec',
             'fiveSigmaDepth', 'visitExposureTime', 'numExposures',
             'visitTime', 'season', 'seeingFwhmEff', 'seeingFwhmGeom',
             'pixRA', 'pixDec']
    types = ['f8']*len(names)
    names += ['night', 'healpixID']
    types += ['i2', 'i2']
    names += ['filter']
    types += ['O']

    daylast = daymin+season_length
    cadence = cadence
    dayobs = np.arange(daymin, daylast, cadence)
    npts = len(dayobs)
    data = np.zeros(npts, dtype=list(zip(names, types)))
    data['observationStartMJD'] = dayobs
    data['night'] = np.floor(data['observationStartMJD']-day0+1)
    data['fiveSigmaDepth'] = m5_ref[band]
    data['visitExposureTime'] = 30.
    data['numExposures'] = 1
    data['visitTime'] = 34
    data['filter'] = band
    data['seeingFwhmEff'] = 0.8
    data['seeingFwhmGeom'] = 0.8
    data['healpixID'] = 10
    data['pixRA'] = 0.0
    data['pixDec'] = 0.0
    return data


def Observations_season(day0=59000, mjdmin=59000, cadence=3.):
    bands = 'grizy'
    Nvisits = dict(zip(bands, [10, 20, 20, 26, 20]))
    rat = 34./3600./24.
    shift_visits = {}
    shift_visits['g'] = 0
    shift_visits['r'] = rat*Nvisits['g']
    shift_visits['i'] = rat*Nvisits['r']
    shift_visits['z'] = rat*Nvisits['i']
    shift_visits['y'] = rat*Nvisits['z']

    # get data
    data = None
    season_length = 180
    shift = 30./(3600.*24)
    for band in bands:

        mjd = mjdmin+shift_visits[band]
        for i in range(Nvisits[band]):
            mjd += shift
            dat = Observations_band(
                daymin=mjd, season_length=season_length, cadence=cadence, band=band)
            if data is None:
                data = dat
            else:
                data = np.concatenate((data, dat))

    return data


def fakeData(band, season=1):

    # Define fake data
    names = ['observationStartMJD', 'fieldRA', 'fieldDec',
             'fiveSigmaDepth', 'visitExposureTime',
             'numExposures', 'visitTime', 'season',
             'seeingFwhmEff', 'seeingFwhmGeom',
             'airmass', 'sky', 'moonPhase', 'pixRA', 'pixDec']

    types = ['f8']*len(names)
    names += ['night']
    types += ['i2']
    names += ['healpixID']
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
    data['night'] = np.floor(data['observationStartMJD']-day0+1)
    data['fiveSigmaDepth'] = m5_ref[band]
    data['visitExposureTime'] = 15.
    data['numExposures'] = 2
    data['visitTime'] = 2.*15.
    data['season'] = season
    data['filter'] = band
    data['seeingFwhmEff'] = 0.
    data['seeingFwhmGeom'] = 0.
    data['airmass'] = 1.2
    data['sky'] = 20.0
    data['moonPhase'] = 0.5
    data['pixRA'] = 0.0
    data['pixDec'] = 0.0
    data['healpixID'] = 1

    return data


class TestSNmetrics(unittest.TestCase):

    def testThroughputs(self):
        """Test the Throughputs class"""
        ## Again, this should not be in MAF but should use appropriate classes in lsst.sims.photUtils.
        # Can we set it up - are the relevant standard environment variables present?
        tp = Throughputs()


    def testTelescope(self):
        """Test the Telescope class"""
        # The Telescope class should be replaced by lsst.sims.photUtils.BandpassSet
        t = Telescope(airmass=1.2)


    def testSNCadenceMetric(self):
        """Test the SN cadence metric """

        # Load up the files from sims_maf_contrib if possible
        sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
        if sims_maf_contrib_dir is not None:
            # Load required SN info to run the metric
            band = 'r'
            SNR = dict(zip('griz', [30., 40., 30., 20.]))  # SNR for WFD
            mag_range = [21., 25.5]  # WFD mag range
            dt_range = [0.5, 30.]  # WFD dt range
            Li_files = [os.path.join(
                sims_maf_contrib_dir, 'data', 'Li_SNCosmo_-2.0_0.2.npy')]
            mag_to_flux_files = [os.path.join(
                sims_maf_contrib_dir, 'data', 'Mag_to_Flux_SNCosmo.npy')]
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
            slicePoint = {'nside': 64}
            metric = SNCadenceMetric(lim_sn=lim_sn, coadd=False)
            result = metric.run(data, slicePoint)

            # And the result should be...
            result_ref = 0.3743514

            assert(np.abs(result-result_ref) < 1.e-5)
        else:
            warnings.warn(
                "skipping SN test because no SIMS_MAF_CONTRIB_DIR set")

    def testSNSNRMetric(self):
        """Test the SN SNR metric """

        sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
        if sims_maf_contrib_dir is not None:
            # Load required SN info to run the metric
            band = 'r'
            z = 0.3
            season = 1.
            Li_files = [os.path.join(
                sims_maf_contrib_dir, 'data', 'Li_SNCosmo_-2.0_0.2.npy')]
            mag_to_flux_files = [os.path.join(
                sims_maf_contrib_dir, 'data', 'Mag_to_Flux_SNCosmo.npy')]

            names_ref = ['SNCosmo']
            coadd = False

            lim_sn = ReferenceData(Li_files, mag_to_flux_files, band, z)

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
            slicePoint = {'nside': 64}
            metric = SNSNRMetric(
                lim_sn=lim_sn, coadd=coadd, names_ref=names_ref, season=season, z=z)

            result = metric.run(data, slicePoint)

            # And the result should be...
            result_ref = 0.4830508474576271

            assert(np.abs(result-result_ref) < 1.e-5)
        else:
            warnings.warn(
                "skipping SN test because no SIMS_MAF_CONTRIB_DIR set")

    def testSNSLMetric(self):
        """Test the SN SL metric """

        # load some fake data
        data = None
        bands = 'griz'
        cadence = dict(zip(bands, [2, 1, 2, 1]))
        for band in bands:
            for i in range(cadence[band]):
                fakes = fakeData(band)
                if data is None:
                    data = fakes
                else:
                    data = np.concatenate((data, fakes))

        # metric instance
        night_collapse = True

        metric = SNSLMetric(night_collapse=night_collapse, nfilters_min=4, min_season_obs=5,
                            m5mins={'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4})

        # run the metric
        nSL = metric.run(data, slicePoint={'nside': 64, 'ra': 0., 'ebv': 0.})

        # and the result should be
        # Changing the refernce value because we have new coadd and mag limits
        nSL_ref = 1.42168e-6  # 0.00012650940

        assert(np.abs(nSL-nSL_ref) < 1.e-8)

    def testNSNMetric(self):
        """ Test the SN NSN metric """
        sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
        if sims_maf_contrib_dir is not None:

            day0 = 59000
            data = None

            diff_season = 280.
            nseasons = 1
            for val in np.arange(59000, 59000+nseasons*diff_season, diff_season):
                dat = Observations_season(day0, val)
                if data is None:
                    data = dat
                else:
                    data = np.concatenate((data, dat))

            #print('data generated', len(data))

            # this is to mimic healpixilization
            nside = 128
            area = hp.nside2pixarea(nside, degrees=True)
            # metric instance
            templateDir = None
            metric = SNNSNMetric(
                pixArea=area, season=[-1], verbose=False, templateDir=templateDir)

            time_ref = time.time()

            res = metric.run(data)

            nSN = res['nSN'].item()
            zlim = res['zlim'].item()

            #print(time.time()-time_ref, nSN, zlim)
            nSN_ref = 2.523
            zlim_ref = 0.65

            assert(np.isclose(nSN, nSN_ref))
            assert(np.isclose(zlim, zlim_ref))
        else:
            warnings.warn(
                "skipping SN test because no SIMS_MAF_CONTRIB_DIR set")

def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
