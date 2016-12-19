import warnings
import numpy as np
import palpy
from lsst.sims.utils import Site, photo_m5

from .baseStacker import BaseStacker

__all__ = ['NormAirmassStacker', 'ParallaxFactorStacker', 'HourAngleStacker',
           'FilterColorStacker', 'ZenithDistStacker', 'ParallacticAngleStacker',
           'SeasonStacker', 'DcrStacker', 'FiveSigmaStacker']

# Original stackers by Peter Yoachim (yoachim@uw.edu)
# Filter color stacker by Lynne Jones (lynnej@uw.edu)
# Season stacker by Phil Marshall (dr.phil.marshall@gmail.com),
# modified by Humna Awan (humna.awan@rutgers.edu)


class FiveSigmaStacker(BaseStacker):
    """
    Calculate the 5-sigma limiting depth for a point source in the given conditions
    """
    def __init__(self, airmassCol='airmass', seeingCol='seeingFwhmEff', skybrightnessCol='skyBrightness',
                 filterCol='filter', exptimeCol='visitExposureTime'):
        self.units = ['mag']
        self.colsAdded = ['fiveSigmaDepth']
        self.colsReq = [airmassCol, seeingCol, skybrightnessCol, filterCol, exptimeCol]
        self.airmassCol = airmassCol
        self.seeingCol = seeingCol
        self.skybrightnessCol = skybrightnessCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol

    def _run(self, simData):
        filts = np.unique(simData[self.filterCol])
        for filtername in filts:
            infilt = np.where(simData[self.filterCol] == filtername)
            simData['fiveSigmaDepth'][infilt] = photo_m5(filtername, simData[infilt][self.skybrightnessCol],
                                                         simData[infilt][self.seeingCol],
                                                         simData[infilt][self.exptimeCol],
                                                         simData[infilt][self.airmassCol])
        return simData


class NormAirmassStacker(BaseStacker):
    """
    Calculate the normalized airmass for each opsim pointing.
    """
    def __init__(self, airmassCol='airmass', decCol='fieldDec',
                 degrees=True, telescope_lat = -30.2446388):

        self.units = ['airmass/(minimum possible airmass)']
        self.colsAdded = ['normairmass']
        self.colsReq = [airmassCol, decCol]
        self.airmassCol = airmassCol
        self.decCol = decCol
        self.telescope_lat = telescope_lat
        self.degrees = degrees

    def _run(self, simData):
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        dec = simData[self.decCol]
        if self.degrees:
            dec = np.radians(dec)
        min_z_possible = np.abs(dec - np.radians(self.telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        simData['normairmass'] = simData[self.airmassCol] / min_airmass_possible
        return simData


class ZenithDistStacker(BaseStacker):
    """
    Calculate the zenith distance for each pointing.
    """
    def __init__(self, altCol = 'altitude', degrees=True):
        self.altCol = altCol
        self.units = ['degrees']
        self.colsAdded = ['zenithDistance']
        self.colsReq = [self.altCol]
        self.degrees = degrees

    def _run(self, simData):
        """Calculate new column for zenith distance."""
        if self.degrees:
            zenithDist = np.pi-np.radians(simData[self.altCol])
        else:
            zenithDist = np.pi-simData[self.altCol]
        simData['zenithDistance'] = np.degrees(zenithDist)
        return simData


class ParallaxFactorStacker(BaseStacker):
    """
    Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='observationStartMJD', raDecDeg=True):
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol
        self.units = ['arcsec', 'arcsec']
        self.colsAdded = ['ra_pi_amp', 'dec_pi_amp']
        self.colsReq = [raCol, decCol, dateCol]
        self.raDecDeg = raDecDeg

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
        ra = simData[self.raCol]
        dec = simData[self.decCol]
        if self.raDecDeg:
            ra = np.radians(ra)
            dec = np.radians(dec)

        for i, ack in enumerate(simData):
            mtoa_params = palpy.mappa(2000., simData[self.dateCol][i])
            # Object with a 1 arcsec parallax
            ra_geo1[i], dec_geo1[i] = palpy.mapqk(ra[i], dec[i],
                                                  0., 0., 1., 0., mtoa_params)
            # Object with no parallax
            ra_geo[i], dec_geo[i] = palpy.mapqk(ra[i], dec[i],
                                                0., 0., 0., 0., mtoa_params)
        x_geo1, y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1,
                                                     ra, dec)
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, ra, dec)
        ra_pi_amp[:] = np.degrees(x_geo1-x_geo)*3600.
        dec_pi_amp[:] = np.degrees(y_geo1-y_geo)*3600.
        simData['ra_pi_amp'] = ra_pi_amp
        simData['dec_pi_amp'] = dec_pi_amp
        return simData


class DcrStacker(BaseStacker):
    """Calculate the RA,Dec offset expected for an object due to differential chromatic refraction.

    Parameters
    ----------
    filterCol : str
        The name of the column with filter names. Default 'fitler'.
    altCol : str
        Name of the column with altitude info. Default 'altitude'.
    raCol : str
        Name of the column with RA. Default 'ra_rad'.
    decCol : str
        Name of the column with Dec. Default 'dec_rad'.
    lstCol : str
        Name of the column with local sidereal time. Default 'lst'.
    site : str or lsst.sims.utils.Site
        Name of the observory or a lsst.sims.utils.Site object. Default 'LSST'.
    mjdCol : str
        Name of column with modified julian date. Default 'observationStartMJD'
    dcr_magnitudes : dict
        Magitude of the DCR offset for each filter at altitude/zenith distance of 45 degrees.
        Defaults u=0.07, g=0.07, r=0.50, i=0.045, z=0.042, y=0.04 (all arcseconds).

    Returns
    -------
    numpy.array
        Returns array with additional columns 'ra_dcr_amp' and 'dec_dcr_amp' with the DCR offsets
        for each observation.  Also runs ZenithDistStacker and ParallacticAngleStacker.
    """

    def __init__(self, filterCol='filter', altCol='altitude', raDecDeg=True,
                 raCol='fieldRA', decCol='fieldDec', lstCol='observationStartLST',
                 site='LSST', mjdCol='observationStartMJD',
                 dcr_magnitudes={'u': 0.07, 'g': 0.07, 'r': 0.050, 'i': 0.045, 'z': 0.042, 'y': 0.04}):

        self.zdCol = 'zenithDistance'
        self.paCol = 'PA'
        self.filterCol = filterCol
        self.raCol = raCol
        self.decCol = decCol
        self.dcr_magnitudes = dcr_magnitudes
        self.colsAdded = ['ra_dcr_amp', 'dec_dcr_amp', 'zenithDistance', 'PA', 'HA']
        self.colsReq = [filterCol, raCol, decCol, altCol, lstCol]
        self.units = ['arcsec', 'arcsec']
        self.raDecDeg = raDecDeg

        self.zstacker = ZenithDistStacker(altCol = altCol)
        self.pastacker = ParallacticAngleStacker(raCol=raCol, decCol=decCol, mjdCol=mjdCol,
                                                 lstCol=lstCol, site=site)

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
    def __init__(self, lstCol='observationStartLST', RaCol='fieldRA'):
        self.units = ['Hours']
        self.colsAdded = ['HA']
        self.colsReq = [lstCol, RaCol]
        self.lstCol = lstCol
        self.RaCol = RaCol

    def _run(self, simData):
        """HA = LST - RA """
        if len(simData) == 0:
            return simData
        ra = np.radians(simData[self.RaCol])
        lst = np.radians(simData[self.lstCol])
        # Check that LST is reasonable
        if (np.min(lst) < 0) | (np.max(lst) > 2.*np.pi):
            warnings.warn('LST values are not between 0 and 2 pi')
        # Check that RA is reasonable
        if (np.min(ra) < 0) | (np.max(ra) > 2.*np.pi):
            warnings.warn('RA values are not between 0 and 2 pi')
        ha = lst - ra
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
    def __init__(self, raCol='fieldRA', decCol='fieldDec', mjdCol='observationStartMJD',
                 lstCol='observationStartLST', site='LSST'):

        self.lstCol = lstCol
        self.raCol = raCol
        self.decCol = decCol
        self.mjdCol = mjdCol

        self.site = Site(name=site)

        self.units = ['radians']
        self.colsAdded = ['PA', 'HA']
        self.colsReq = [self.raCol, self.decCol, self.mjdCol, self.lstCol]
        self.haStacker = HourAngleStacker(lstCol=lstCol, RaCol=raCol)

    def _run(self, simData):
        # Equation from:
        # http://www.gb.nrao.edu/~rcreager/GBTMetrology/140ft/l0058/gbtmemo52/memo52.html
        # or
        # http://www.gb.nrao.edu/GBT/DA/gbtidl/release2pt9/contrib/contrib/parangle.pro
        simData = self.haStacker._run(simData)
        dec = np.radians(simData[self.decCol])
        simData['PA'] = np.arctan2(np.sin(simData['HA']*np.pi/12.), (np.cos(dec) *
                                   np.tan(self.site.latitude_rad) - np.sin(dec) *
                                   np.cos(simData['HA']*np.pi/12.)))
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
    def __init__(self, observationStartMJDCol='observationStartMJD', RACol='fieldRA'):
        # Names of columns we want to add.
        self.colsAdded = ['year', 'season']
        # Names of columns we need from database.
        self.colsReq = [observationStartMJDCol, RACol]
        # List of units for our new columns.
        self.units = ['', '']
        # And save the column names.
        self.observationStartMJDCol = observationStartMJDCol
        self.RACol = RACol

    def _run(self, simData):
        # Define year number: (note that opsim defines "years" in flat 365 days).
        year = np.floor((simData[self.observationStartMJDCol] - simData[self.observationStartMJDCol][0]) / 365)
        objRA = simData[self.RACol]/15.0   # in hrs
        # objRA=0 on autumnal equinox.
        # autumnal equinox 2014 happened on Sept 23 --> Equinox MJD
        Equinox = 2456923.5 - 2400000.5
        # Use 365.25 for the length of a year here, because we're dealing with real seasons.
        daysSinceEquinox = 0.5*objRA*(365.25/12.0)  # 0.5 to go from RA to month; 365.25/12.0 months to days
        firstSeasonBegan = Equinox + daysSinceEquinox - 0.5*365.25   # in MJD
        # Now we can compute the number of years since the first season
        # began, and so assign a global integer season number:
        globalSeason = np.floor((simData[self.observationStartMJDCol] - firstSeasonBegan)/365.25)
        # Subtract off season number of first observation:
        season = globalSeason - np.min(globalSeason)
        simData['year'] = year
        simData['season'] = season
        return simData

