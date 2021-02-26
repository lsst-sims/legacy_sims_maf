import numpy as np
import lsst.sims.maf.metrics as metrics
import healpy as hp
from .seasonMetrics import calcSeason
from scipy.stats import binned_statistic
from lsst.sims.utils import int_binned_stat
from lsst.sims.photUtils import Dust_values

__all__ = ['SNSLMetric']


def coaddM5(mags):
    """Coadded depth, assuming Gaussian noise
    """
    return 1.25 * np.log10(np.sum(10.**(0.8*np.array(mags))))


class SNSLMetric(metrics.BaseMetric):
    def __init__(self, metricName='SNSLMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', m5Col='fiveSigmaDepth', season=[-1], night_collapse=False,
                 nfilters_min=4, min_season_obs=5,
                 # XXX--I just made up some reasonable sounding numbers
                 m5mins={'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}, 
                 maps=['DustMap'], **kwargs):
        """
        Strongly Lensed SN metric

        The number of is given by:

        N (lensed SNe Ia with well measured time delay) = 45.7 * survey_area /
        (20000 deg^2) * cumulative_season_length / (2.5 years) / (2.15 *
        exp(0.37 * gap_median_all_filter))

        where:
        survey_area: survey area (in deg2)
        cumulative_season_length: cumulative season length (in years)
        gap_median_all_filter: median gap (all filters)

        Parameters
        --------------
        metricName : str, opt
         metric name
         Default : SNCadenceMetric
        mjdCol : str, opt
         mjd column name
         Default : observationStartMJD,
        RaCol : str,opt
         Right Ascension column name
         Default : fieldRa
        DecCol : str,opt
         Declinaison column name
         Default : fieldDec
        filterCol : str,opt
         filter column name
         Default: filter
        exptimeCol : str,opt
         exposure time column name
         Default : visitExposureTime
        nightCol : str,opt
         night column name
         Default : night
        obsidCol : str,opt
         observation id column name
         Default : observationId
        nexpCol : str,opt
         number of exposure column name
         Default : numExposures
        vistimeCol : str,opt
         visit time column name
         Default : visitTime
        season: int (list) or -1, opt
         season to process (default: -1: all seasons)
        nfilters_min : int (5)
            The number of filters to demand in a season


        """
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seasonCol = 'season'
        self.m5Col = m5Col
        self.maps= maps

        cols = [self.nightCol, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.m5Col]

        super(SNSLMetric, self).__init__(
            col=cols, metricName=metricName, maps=self.maps, units='N SL', **kwargs)
        self.badVal = 0
        self.season = season
        self.bands = 'ugrizy'

        self.night_collapse = night_collapse
        self.m5mins = m5mins
        self.min_season_obs = min_season_obs
        self.nfilters_min = nfilters_min
        self.phot_properties = Dust_values()

    def run(self, dataSlice, slicePoint=None):
        """
        Runs the metric for each dataSlice

        Parameters
        ---------------
        dataSlice: simulation data
        slicePoint:  slicePoint(default None)

        Returns
        -----------
        number of SL time delay supernovae

        """
        dataSlice.sort(order=self.mjdCol)

        # Crop it down so things are coadded per night at the median MJD time
        dataSlice = self.coadd(dataSlice)

        # get the pixel area
        area = hp.nside2pixarea(slicePoint['nside'], degrees=True)

        if len(dataSlice) == 0:
            return self.badVal

        season_id = np.floor(calcSeason(np.degrees(slicePoint['ra']), dataSlice[self.mjdCol]))

        seasons = self.season

        if self.season == [-1]:
            seasons = np.unique(season_id)

        season_lengths = []
        median_gaps = []
        for season in seasons:
            idx = np.where(season_id == season)[0]
            bright_enough = np.zeros(idx.size, dtype=bool)
            for key in self.m5mins:
                in_filt = np.where(dataSlice[idx][self.filterCol] == key)[0]
                A_x = self.phot_properties.Ax1[key] * slicePoint['ebv']
                bright_enough[in_filt[np.where((dataSlice[idx[in_filt]][self.m5Col] - A_x) > self.m5mins[key])[0]]] = True
            idx = idx[bright_enough]
            u_filters = np.unique(dataSlice[idx][self.filterCol])
            if (len(idx) < self.min_season_obs) | (np.size(u_filters) < self.nfilters_min):
                continue
            if self.night_collapse:
                u_nights, unight_indx = np.unique(dataSlice[idx][self.nightCol], return_index=True)
                idx = idx[unight_indx]
                order = np.argsort(dataSlice[self.mjdCol][idx])
                idx = idx[order]
            mjds_season = dataSlice[self.mjdCol][idx]
            cadence = mjds_season[1:]-mjds_season[:-1]
            season_lengths.append(mjds_season[-1]-mjds_season[0])
            median_gaps.append(np.median(cadence))

        # get the cumulative season length

        cumul_season_length = np.sum(season_lengths)

        if cumul_season_length == 0:
            return self.badVal
        # get gaps
        gap_median = np.mean(median_gaps)

        # estimate the number of lensed supernovae
        cumul_season = cumul_season_length/(12.*30.)

        N_lensed_SNe_Ia = 45.7 * area / 20000. * cumul_season /\
            2.5 / (2.15 * np.exp(0.37 * gap_median))

        return N_lensed_SNe_Ia

    def coadd(self, dataSlice):
        """
        Method to coadd data per band and per night

        Parameters
        ---------------
        dataSlice : np.array

        Returns
        -----------
        dataSlice, with m5Col replaced with m5 coadded in the night

        """

        dataSlice.sort(order=[self.nightCol, self.filterCol])

        filters = np.unique(dataSlice[self.filterCol])

        for filtername in filters:
            infilt = np.where(dataSlice[self.filterCol] == filtername)[0]
            unight, indices = np.unique(dataSlice[self.nightCol][infilt], return_inverse=True)
            right = unight+0.5
            bins = [unight[0]-0.5] + right.tolist()
            coadds, be, bn = binned_statistic(dataSlice[self.nightCol][infilt],
                                              dataSlice[self.m5Col][infilt], bins=bins,
                                              statistic=coaddM5)
            # now we just clobber the m5Col with the coadd for the night
            dataSlice[self.m5Col][infilt] = coadds[indices]

            unights, median_mjd_per_night = int_binned_stat(dataSlice[self.nightCol][infilt],
                                                            dataSlice[self.mjdCol][infilt],
                                                            statistic=np.median)
            dataSlice[self.mjdCol][infilt] = median_mjd_per_night[indices]

            # Note, we could also clobber exposure MJD, exposure time, etc. But I don't think they
            # get used later, so maybe OK.

        # Make a string that is the combination of night and filter
        night_filt = np.char.add(dataSlice[self.nightCol].astype(str), dataSlice[self.filterCol].astype(str))
        u_nf, indx = np.unique(night_filt, return_index=True)
        dataSlice = dataSlice[indx]
        dataSlice.sort(order=[self.mjdCol])

        return dataSlice
