import warnings
import numpy as np
import palpy
from lsst.sims.utils import _altAzPaFromRaDec, ObservationMetaData, Site

from .baseStacker import BaseStacker

__all__ = ['NormAirmassStacker', 'ParallaxFactorStacker', 'HourAngleStacker',
           'FilterColorStacker', 'ZenithDistStacker', 'ParallacticAngleStacker',
           'SeasonStacker', 'DcrStacker']

# Original stackers by Peter Yoachim (yoachim@uw.edu)
# Filter color stacker by Lynne Jones (lynnej@uw.edu)
# Season stacker by Phil Marshall (dr.phil.marshall@gmail.com),
# modified by Humna Awan (humna.awan@rutgers.edu)


class NormAirmassStacker(BaseStacker):
    """
    Calculate the normalized airmass for each opsim pointing.
    """
    def __init__(self, airmassCol='airmass', decCol='fieldDec', telescope_lat = -30.2446388):

        self.units = ['airmass/(minimum possible airmass)']
        self.colsAdded = ['normairmass']
        self.colsReq = [airmassCol, decCol]
        self.airmassCol = airmassCol
        self.decCol = decCol
        self.telescope_lat = telescope_lat

    def _run(self, simData):
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        min_z_possible = np.abs(simData[self.decCol] - np.radians(self.telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        simData['normairmass'] = simData[self.airmassCol] / min_airmass_possible
        return simData


class ZenithDistStacker(BaseStacker):
    """
    Calculate the zenith distance for each pointing.
    """
    def __init__(self, altCol = 'altitude'):
        self.altCol = altCol
        self.units = ['radians']
        self.colsAdded = ['zenithDistance']
        self.colsReq = [self.altCol]

    def _run(self, simData):
        """Calculate new column for zenith distance."""
        zenithDist = np.pi-simData[self.altCol]
        simData['zenithDistance'] = zenithDist
        return simData


class ParallaxFactorStacker(BaseStacker):
    """
    Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='expMJD'):
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol
        self.units = ['arcsec', 'arcsec']
        self.colsAdded = ['ra_pi_amp', 'dec_pi_amp']
        self.colsReq = [raCol, decCol, dateCol]

    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """
        Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians.
        """
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y

    def _run(self, simData):
        ra_pi_amp = np.zeros(np.size(simData), dtype=[('ra_pi_amp', 'float')])
        dec_pi_amp = np.zeros(np.size(simData), dtype=[('dec_pi_amp', 'float')])
        ra_geo1 = np.zeros(np.size(simData), dtype='float')
        dec_geo1 = np.zeros(np.size(simData), dtype='float')
        ra_geo = np.zeros(np.size(simData), dtype='float')
        dec_geo = np.zeros(np.size(simData), dtype='float')
        for i, ack in enumerate(simData):
            mtoa_params = palpy.mappa(2000., simData[self.dateCol][i])
            # Object with a 1 arcsec parallax
            ra_geo1[i], dec_geo1[i] = palpy.mapqk(simData[self.raCol][i], simData[self.decCol][i],
                                                  0., 0., 1., 0., mtoa_params)
            # Object with no parallax
            ra_geo[i], dec_geo[i] = palpy.mapqk(simData[self.raCol][i], simData[self.decCol][i],
                                                0., 0., 0., 0., mtoa_params)
        x_geo1, y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1,
                                                     simData[self.raCol], simData[self.decCol])
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, simData[self.raCol], simData[self.decCol])
        ra_pi_amp[:] = np.degrees(x_geo1-x_geo)*3600.
        dec_pi_amp[:] = np.degrees(y_geo1-y_geo)*3600.
        simData['ra_pi_amp'] = ra_pi_amp
        simData['dec_pi_amp'] = dec_pi_amp
        return simData


class DcrStacker(BaseStacker):
    """
    Similar to the parallax stacker, calculate the x,y offset in the gnomic
    projection for an object based on DCR.
    """

    def __init__(self, zdCol='zenithDistance', filterCol='filter',
                 raCol='fieldRA', decCol='fieldDec', paCol='PA',
                 dcr_magnitudes={'u': 0.07, 'g': 0.07, 'r': 0.050, 'i': 0.045, 'z': 0.042, 'y': 0.04}):
        self.zdCol = zdCol
        self.paCol = paCol
        self.filterCol = filterCol
        self.raCol = raCol
        self.decCol = decCol
        self.dcr_magnitudes = dcr_magnitudes
        self.colsAdded = ['ra_dcr_amp', 'dec_dcr_amp', 'zenithDistance', 'PA']
        self.colsReq = [filterCol, raCol, decCol, 'altitude']
        self.units = ['arcsec', 'arcsec']

        self.zstacker = ZenithDistStacker()
        self.pastacker = ParallacticAngleStacker()

    def _run(self, simData):
        # Need to make sure the Zenith stacker gets run first
        simData = self.zstacker._run(simData)
        simData = self.pastacker._run(simData)

        dcr_in_ra = np.tan(simData[self.zdCol])*np.sin(simData[self.paCol])
        dcr_in_dec = np.tan(simData[self.zdCol])*np.cos(simData[self.paCol])
        for filtername in np.unique(simData[self.filterCol]):
            fmatch = np.where(simData[self.filterCol] == filtername)
            dcr_in_ra[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_ra[fmatch]
            dcr_in_dec[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_dec[fmatch]
        simData['ra_dcr_amp'] = dcr_in_ra
        simData['dec_dcr_amp'] = dcr_in_dec

        return simData


class HourAngleStacker(BaseStacker):
    """
    Add the Hour Angle for each observation.
    """
    def __init__(self, lstCol='lst', RaCol='fieldRA'):
        self.units = ['Hours']
        self.colsAdded = ['HA']
        self.colsReq = [lstCol, RaCol]
        self.lstCol = lstCol
        self.RaCol = RaCol

    def _run(self, simData):
        """HA = LST - RA """
        if len(simData) == 0:
            return simData
        # Check that LST is reasonable
        if (np.min(simData[self.lstCol]) < 0) | (np.max(simData[self.lstCol]) > 2.*np.pi):
            warnings.warn('LST values are not between 0 and 2 pi')
        # Check that RA is reasonable
        if (np.min(simData[self.RaCol]) < 0) | (np.max(simData[self.RaCol]) > 2.*np.pi):
            warnings.warn('RA values are not between 0 and 2 pi')
        ha = simData[self.lstCol] - simData[self.RaCol]
        # Wrap the results so HA between -pi and pi
        ha = np.where(ha < -np.pi, ha+2.*np.pi, ha)
        ha = np.where(ha > np.pi, ha-2.*np.pi, ha)
        # Convert radians to hours
        simData['HA'] = ha*12/np.pi
        return simData


class ParallacticAngleStacker(BaseStacker):
    """
    Add the parallactic angle (in radians) to each visit.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', mjdCol='expMJD', latRad=None,
                 lonRad=None, height=None, tempCentigrade=None, lapseRate=None,
                 humidity=None, pressure=None):

        self.raCol = raCol
        self.decCol = decCol
        self.mjdCol = mjdCol

        if latRad is None:
            latDeg = None
        else:
            latDeg = np.degrees(latRad)

        if lonRad is None:
            lonDeg = None
        else:
            lonDeg = np.degrees(lonRad)

        self.site = Site(longitude=lonDeg, latitude=latDeg,
                         temperature=tempCentigrade,
                         height=height, humidity=humidity,
                         pressure=pressure, lapseRate=lapseRate,
                         name='LSST')

        self.units = ['radians']
        self.colsAdded = ['PA']
        self.colsReq = [self.raCol, self.decCol, self.mjdCol]

    def _run(self, simData):
        pa_arr = []
        for ra, dec, mjd in zip(simData[self.raCol], simData[self.decCol], simData[self.mjdCol]):
            # Catch time warnings since we don't have future leap seconds
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alt, az, pa = _altAzPaFromRaDec(ra, dec,
                                                ObservationMetaData(mjd=mjd, site=self.site))

            pa_arr.append(pa)

        simData['PA'] = np.array(pa_arr)
        return simData


class FilterColorStacker(BaseStacker):
    """
    Translate filters ('u', 'g', 'r' ..) into RGB tuples.
    """
    def __init__(self, filterCol='filter', filterMap={'u': 1, 'g': 2, 'r': 3, 'i': 4, 'z': 5, 'y': 6}):
        self.filter_rgb_map = {'u': (0, 0, 1),   # dark blue
                               'g': (0, 1, 1),  # cyan
                               'r': (0, 1, 0),    # green
                               'i': (1, 0.5, 0.3),  # orange
                               'z': (1, 0, 0),    # red
                               'y': (1, 0, 1)}  # magenta
        self.filterCol = filterCol
        # self.units used for plot labels
        self.units = ['rChan', 'gChan', 'bChan']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['rRGB', 'gRGB', 'bRGB']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.filterCol]

    def _run(self, simData):
        # Translate filter names into numbers.
        filtersUsed = np.unique(simData[self.filterCol])
        for f in filtersUsed:
            if f not in self.filter_rgb_map:
                raise IndexError('Filter %s not in filter_rgb_map' % (f))
            match = np.where(simData[self.filterCol] == f)[0]
            simData['rRGB'][match] = self.filter_rgb_map[f][0]
            simData['gRGB'][match] = self.filter_rgb_map[f][1]
            simData['bRGB'][match] = self.filter_rgb_map[f][2]
        return simData


class SeasonStacker(BaseStacker):
    """
    Add an integer label to show which season a given visit is in.
    The season only depends on the RA of the object: we compute the MJD
    when each object is on the meridian at midnight, and subtract 6
    months to get the start date of each season.
    The season index range is 0-10.
    Must wrap 0th and 10th to get a total of 10 seasons.
    """
    def __init__(self, expMJDCol='expMJD', RACol='fieldRA'):
        # Names of columns we want to add.
        self.colsAdded = ['year', 'season']
        # Names of columns we need from database.
        self.colsReq = [expMJDCol, RACol]
        # List of units for our new columns.
        self.units = ['', '']
        # And save the column names.
        self.expMJDCol = expMJDCol
        self.RACol = RACol

    def _run(self, simData):
        # Define year number: (note that opsim defines "years" in flat 365 days).
        year = np.floor((simData[self.expMJDCol] - simData[self.expMJDCol][0]) / 365)
        objRA = np.degrees(simData[self.RACol])/15.0   # in hrs
        # objRA=0 on autumnal equinox.
        # autumnal equinox 2014 happened on Sept 23 --> Equinox MJD
        Equinox = 2456923.5 - 2400000.5
        # Use 365.25 for the length of a year here, because we're dealing with real seasons.
        daysSinceEquinox = 0.5*objRA*(365.25/12.0)  # 0.5 to go from RA to month; 365.25/12.0 months to days
        firstSeasonBegan = Equinox + daysSinceEquinox - 0.5*365.25   # in MJD
        # Now we can compute the number of years since the first season
        # began, and so assign a global integer season number:
        globalSeason = np.floor((simData[self.expMJDCol] - firstSeasonBegan)/365.25)
        # Subtract off season number of first observation:
        season = globalSeason - np.min(globalSeason)
        simData['year'] = year
        simData['season'] = season
        return simData

