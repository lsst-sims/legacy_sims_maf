import numpy as np
from .baseStacker import BaseStacker


__all__ = ['NEODistStacker']

class NEODistStacker(BaseStacker):
    """
    For each observation, find the max distance to a 144 km NEO, also stack on the x,y position of the asteroid
    """

    def __init__(self, m5Col='fiveSigmaDepth',
                 stepsize=.001, maxDist=2.5,H=22, elongCol='solarElong', filterCol='filter',**kwargs):

        """
        stepsize:  The stepsize to use when solving (in AU)
        maxDist: How far out to try and measure (in AU)
        """

        self.units = ['AU','AU','AU']
        self.colsReq=[elongCol, filterCol,m5Col]
        self.colsAdded=['NEODist', 'NEOX','NEOY']

        self.m5Col= m5Col
        self.elongCol = elongCol
        self.filterCol = filterCol

        self.H = H
        # Magic numbers that convert an asteroid V-band magnitude to LSST filters:
        # V_5 = m_5 + (adjust value)
        self.limitingAdjust = {'u':-2.1, 'g':-0.5,'r':0.2,'i':0.4,'z':0.6,'y':0.6}
        self.deltas = np.arange(stepsize,maxDist+stepsize,stepsize)
        self.G = 0.15

        # magic numbers from  http://adsabs.harvard.edu/abs/2002AJ....124.1776J
        self.a1 = -3.33
        self.b1 = 0.63
        self.a2 = 1.87
        self.b2 = 1.22


    def run(self,simData, slicePoint=None):

        simData=self._addStackers(simData)

        elongRad = np.radians(simData[self.elongCol])

        v5 = np.zeros(simData.size, dtype=float) + simData[self.m5Col]
        for filterName in self.limitingAdjust:
            fmatch = np.where(simData[self.filterCol] == filterName)
            v5[fmatch] += self.limitingAdjust[filterName]

        for i,elong in enumerate(elongRad):
            # Law of cosines:
            # Heliocentric Radius of the object
            R = np.sqrt(1.+self.deltas**2-2.*self.deltas*np.cos(elong) )
            # Angle between sun and earth as seen by NEO
            alphas = np.arccos( (1.-R**2-self.deltas**2)/(-2.*self.deltas*R) )
            ta2 = np.tan(alphas/2.)
            phi1 = np.exp(-self.a1*ta2**self.b1)
            phi2 = np.exp(-self.a2*ta2**self.b2)

            alpha_term = 2.5*np.log10( (1.- self.G)*phi1+self.G*phi2)
            appmag = self.H+5.*np.log10(R*self.deltas)-alpha_term
            good = np.where(appmag < v5[i])
            simData['NEODist'][i] = np.max(self.deltas[good])

        simData['NEOX'] = simData['NEODist']*np.cos(elongRad-np.pi/2.)
        simData['NEOY'] = simData['NEODist']*np.sin(elongRad-np.pi/2.)

        return simData
