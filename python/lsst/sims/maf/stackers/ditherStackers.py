import numpy as np
import numpy.lib.recfunctions as rfn
from .baseStacker import BaseStacker
        

# Add a new dither pattern
class DecOnlyDitherStacker(BaseStacker):
    """Dither the position of pointings in dec only.  """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night',
                 nightStep=1, nSteps=5, stepSize=0.2):
        """stepsize in degrees """
        self.raCol = raCol
        self.decCol = decCol
        self.nightCol = nightCol
        self.nightStep=nightStep
        self.nSteps = nSteps
        self.stepSize = stepSize
        self.units = 'rad'
        self.colsAdded = ['decOnlyDither']
        self.colsReq =[raCol, decCol, nightCol]


    def run(self, simData):
        off1 = np.arange(self.nSteps+1)*self.stepSize
        off2 = off1[::-1][1:]
        off3 = -1.*off1[1:]
        off4 = off3[::-1][1:]
        offsets = np.radians(np.concatenate((off1,off2,off3,off4) ))
        uoffsets = np.size(offsets)
        nightIndex = simData[self.nightCol]%uoffsets
        simData= self._addStackers(simData)
        simData['decOnlyDither'] = simData[self.decCol]+offsets[nightIndex]
        
        return simData
    
                             
