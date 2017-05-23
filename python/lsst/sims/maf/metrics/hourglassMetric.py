from builtins import zip
import numpy as np
from .baseMetric import BaseMetric
from lsst.sims.utils import Site

__all__ = ['HourglassMetric']


def nearestVal(A, val):
    return A[np.argmin(np.abs(np.array(A)-val))]


class HourglassMetric(BaseMetric):
    """Plot the filters used as a function of time.
    Must be used with the Hourglass Slicer.
    """
    def __init__(self, telescope='LSST', **kwargs):

        metricName = 'hourglass'
        filtercol = "filter"
        mjdcol = "expMJD"
        nightcol = "night"
        cols = [filtercol, mjdcol, nightcol]
        super(HourglassMetric, self).__init__(col=cols, metricName=metricName, metricDtype='object', **kwargs)
        self.nightcol = nightcol
        self.mjdcol = mjdcol
        self.filtercol = filtercol
        self.telescope = Site(name=telescope)

    def run(self, dataSlice, slicePoint=None):

        import ephem
        dataSlice.sort(order=self.mjdcol)
        unights, uindx = np.unique(dataSlice[self.nightcol], return_index=True)

        names = ['mjd', 'midnight', 'moonPer', 'twi6_rise', 'twi6_set', 'twi12_rise',
                 'twi12_set', 'twi18_rise', 'twi18_set']
        types = ['float']*len(names)
        pernight = np.zeros(len(unights), dtype=list(zip(names, types)))
        pernight['mjd'] = dataSlice['expMJD'][uindx]

        left = np.searchsorted(dataSlice[self.nightcol], unights)

        lsstObs = ephem.Observer()
        lsstObs.lat = self.telescope.latitude_rad
        lsstObs.lon = self.telescope.longitude_rad
        lsstObs.elevation = self.telescope.height
        horizons = ['-6', '-12', '-18']
        key = ['twi6', 'twi12', 'twi18']

        obsList = []
        S = ephem.Sun()
        moon = ephem.Moon()
        for h in horizons:
            obs = ephem.Observer()
            obs.lat, obs.lon, obs.elevation = self.telescope.latitude_rad, \
                                              self.telescope.longitude_rad, self.telescope.height
            obs.horizon = h
            obsList.append(obs)

        # Oh for fuck's sake...pyephem uses 1899 as it's zero-day, and MJD has Nov 17 1858 as zero-day.
        doff = ephem.Date(0)-ephem.Date('1858/11/17')

        for i, mjd in enumerate(pernight['mjd']):
            mjd = mjd-doff

            pernight['midnight'][i] = nearestVal([lsstObs.previous_antitransit(S, start=mjd),
                                                  lsstObs.next_antitransit(S, start=mjd)], mjd) + doff
            moon.compute(mjd)
            pernight['moonPer'][i] = moon.phase
            for j, obs in enumerate(obsList):
                pernight[key[j]+'_rise'][i] = obs.next_rising(S, start=pernight['midnight'][i]-doff,
                                                              use_center=True) + doff

                pernight[key[j]+'_set'][i] = obs.previous_setting(S, start=pernight['midnight'][i]-doff,
                                                                  use_center=True) + doff

        # Define the breakpoints as where either the filter changes
        # OR there's more than a 2 minute gap in observing
        good = np.where((dataSlice[self.filtercol] != np.roll(dataSlice[self.filtercol], 1)) |
                        (np.abs(np.roll(dataSlice[self.mjdcol], 1) -
                                dataSlice[self.mjdcol]) > 120. / 3600. / 24.))[0]
        good = np.concatenate((good, [0], [len(dataSlice[self.filtercol])]))
        good = np.unique(good)
        left = good[:-1]
        right = good[1:]-1
        good = np.ravel(list(zip(left, right)))

        names = ['mjd', 'midnight', 'filter']
        types = ['float', 'float', '|S1']
        perfilter = np.zeros((good.size), dtype=list(zip(names, types)))
        perfilter['mjd'] = dataSlice['expMJD'][good]
        perfilter['filter'] = dataSlice['filter'][good]
        for i, mjd in enumerate(perfilter['mjd']):
            mjd = mjd - doff
            perfilter['midnight'][i] = nearestVal([lsstObs.previous_antitransit(S, start=mjd),
                                                   lsstObs.next_antitransit(S, start=mjd)], mjd)+doff

        return {'pernight': pernight, 'perfilter': perfilter}
