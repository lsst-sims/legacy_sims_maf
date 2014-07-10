import numpy as np
from .baseStacker import BaseStacker
        
def wrapRA(ra):
    """
    Wrap RA values so they are between 0 and 2pi (using mod).
     """
    ra = ra % (2.0*np.pi)
    return ra

def wrapDec(dec):
    """
    Wrap dec positions to be between -pi/2 and pi/2.
    (reflects Dec values around +/- 90).
    """
    dec = np.where(dec < -np.pi/2.0, -1.*(np.pi + dec), dec)
    dec = np.where(dec > np.pi/2.0, (np.pi - dec), dec)
    return dec


class RandomDitherStacker(BaseStacker):
    """Randomly dither the RA and Dec pointings up to maxDither degrees from center."""
    def __init__(self, raCol='fieldRA', decCol='fieldDec', maxDither=1.8, randomSeed=None):
        # Instantiate the RandomDither object and set internal variables. 
        self.raCol = raCol
        self.decCol = decCol
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
        # Generate the random dither values.
        dithersRA = np.random.rand(len(simData[self.raCol]))
        dithersDec = np.random.rand(len(simData[self.decCol]))
        # np.random.rand returns numbers in [0, 1) interval.
        # Scale to desired +/- maxDither range.
        dithersRA = dithersRA*np.cos(simData[self.decCol])*2.0*self.maxDither - self.maxDither
        dithersDec = dithersDec*2.0*self.maxDither - self.maxDither
        # Add columns to simData and then fill with new values.
        simData = self._addStackers(simData)
        # Add to RA and wrap back into expected range.
        simData['randomRADither'] = wrapRA(simData[self.raCol] + dithersRA)
        # Add to Dec and wrap back into expected range.
        simData['randomDecDither'] = wrapDec(simData[self.decCol] + dithersDec)
        return simData


# Add a new dither pattern (sily example)
class DecOnlyDitherStacker(BaseStacker):
    """Dither the position of pointings in dec only.  """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night',
                 nightStep=1, nSteps=5, stepSize=0.2):
        """stepsize in degrees """
        self.raCol = raCol
        self.decCol = decCol
        self.nightCol = nightCol
        self.nightStep = nightStep
        self.nSteps = nSteps
        self.stepSize = stepSize
        self.units = ['rad']
        self.colsAdded = ['decOnlyDither']
        self.colsReq = [raCol, decCol, nightCol]

    def run(self, simData):
        off1 = np.arange(self.nSteps+1)*self.stepSize
        off2 = off1[::-1][1:]
        off3 = -1.*off1[1:]
        off4 = off3[::-1][1:]
        offsets = np.radians(np.concatenate((off1,off2,off3,off4) ))
        uoffsets = np.size(offsets)
        nightIndex = simData[self.nightCol]%uoffsets
        simData= self._addStackers(simData)
        simData['decOnlyDither'] = wrapDec(simData[self.decCol]+offsets[nightIndex])        
        return simData
    
                             
