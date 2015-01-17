import numpy as np
from .baseStacker import BaseStacker

def wrapRADec(ra, dec):
    """
    Wrap RA and Dec values so RA between 0-2pi (using mod),
      and Dec in +/- pi/2.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi/2.0)[0]
    dec[low] = -1.*(np.pi + dec)
    ra[low] = ra - np.pi
    high = np.where(dec > np.pi/2.0)[0]
    dec[high] = np.pi - dec
    ra[high] = ra - np.pi
    # Wrap RA.
    ra = ra % (2.0*np.pi)
    return ra, dec

def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).
    """
    ra = ra % (2.0*np.pi)
    return ra

class RandomDitherStacker(BaseStacker):
    """Randomly dither the RA and Dec pointings up to maxDither degrees from center, per pointing."""
    def __init__(self, raCol='fieldRA', decCol='fieldDec', maxDither=1.8, randomSeed=None):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = maxDither * np.pi / 180.0
        self.randomSeed = randomSeed
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['randomRADither', 'randomDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol]

    def run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Add new columns to simData, ready to fill with new values.
        simData = self._addStackers(simData)
        # Generate the random dither values.
        nobs = len(simData[self.raCol])
        dithersRA = (np.random.rand(nobs)*2.0*self.maxDither - self.maxDither)*np.cos(simData[self.decCol])
        dithersDec = np.random.rand(nobs)*2.0*self.maxDither - self.maxDither
        # Add to RA and dec values.
        simData['randomRADither'] = simData[self.raCol] + dithersRA
        simData['randomDecDither'] = simData[self.decCol] + dithersDec
        # Wrap back into expected range.
        simData['randomRADither'], simData['randomDecDither'] = wrapRADec(simData['randomRADither'], simData['randomDecDither'])
        return simData

class NightlyRandomDitherStacker(BaseStacker):
    """Randomly dither the RA and Dec pointings up to maxDither degrees from center, one dither offset per night."""
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night', maxDither=1.8, randomSeed=None):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        self.nightCol = nightCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = maxDither * np.pi / 180.0
        self.randomSeed = randomSeed
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['nightlyRandomRADither', 'nightlyRandomDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol]

    def run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Add the new columns to simData.
        simData = self._addStackers(simData)
        # Generate the random dither values, one per night.
        nights = np.unique(simData[self.nightCol])
        nightDithersRA = np.random.rand(len(nights))*2.0*self.maxDither - self.maxDither
        nightDithersDec = np.random.rand(len(nights))*2.0*self.maxDither - self.maxDither
        for n, dra, ddec in zip(nights, nightDithersRA, nightDithersDec):
            match = np.where(simData[self.nightCol] == n)[0]
            simData['nightlyRandomRADither'][match] = simData[self.raCol][match] + dra*np.cos(simData[self.decCol][match])
            simData['nightlyRandomDecDither'][match] = simData[self.decCol][match] + ddec
        # Wrap RA/Dec into expected range.
        simData['nightlyRandomRADither'], simData['nightlyRandomDecDither'] = wrapRADec(simData['nightlyRandomRADither'],
                                                                                        simData['nightlyRandomDecDither'])
        return simData


