import numpy as np
from .baseStacker import BaseStacker
from lsst.sims.utils import galacticFromEquatorial

__all__ = ['GalacticStacker']

class GalacticStacker(BaseStacker):
    """
    Stack on the galactic coordinates of each pointing.
    """
    def __init__(self, mjdCol='expMJD', raCol='fieldRA',decCol='fieldDec'):

        self.colsReq = [raCol,decCol]
        self.colsAdded = ['gall','galb']
        self.units = ['radians', 'radians']
        self.raCol = raCol
        self.decCol=decCol

    def run(self,simData):
        simData=self._addStackers(simData)
        simData['gall'], simData['galb'] = galacticFromEquatorial(simData[self.raCol], simData[self.decCol])
        return simData
