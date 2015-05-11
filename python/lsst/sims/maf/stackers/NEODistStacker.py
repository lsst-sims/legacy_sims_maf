import numpy as np
from .baseStacker import BaseStacker


__all__ = ['NEODistStacker']

class NEODistStacker(BaseStacker):
    """
    For each observation, find the max distance to a 144 km NEO, also stack on the x,y position of the asteroid
    """

    def __init__(self, m5Col='fiveSigmaDepth',
                 stepsize=.001, maxDist=2.,H=22, elongCol='solarElong', filterCol='filter',**kwargs):

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
        stepsize

        # Magic numbers that convert an asteroid V-band magnitude to LSST filters:
        # V_5 = m_5 + (adjust value)
        self.limitingAdjust = {'u':-2.1, 'g':-0.5,'r':0.2,'i':0.4,'z':0.6,'y':0.6}
        self.deltas = np.arange(stepsize,maxDist+stepsize,stepsize)
        self.G = 0.15


    # Phi approximations from:
    # http://adsabs.harvard.edu/abs/2002AJ....124.1776J
    def phi1(self,alpha):
        result = np.exp( -3.33*np.tan(0.5*alpha)**0.63)
        return result
    def phi2(self,alpha):
        result = np.exp( 1.87*np.tan(0.5*alpha)**1.22)
        return result


    def run(self,simData, slicePoint=None):

        simData=self._addStackers(simData)

        v5 = np.zeros(simData.size, dtype=float) + simData[self.m5Col]
        for filterName in self.limitingAdjust:
            good = np.where(simData[self.filterCol] == filterName)
            v5[good] += self.limitingAdjust[filterName]

        for i,elong in enumerate(simData[self.elongCol]):
            # Law of cosines:
            R = np.sqrt(1.+self.deltas**2-2.*self.deltas*np.cos(elong) )
            alphas = np.arccos( (1.-R**2-self.deltas**2)/(-2.*self.deltas*R) )
            alpha_term = 2.5*np.log( (1.- self.G)*self.phi1(alphas)+self.G*self.phi2(alphas))
            appmag = self.H+5.*np.log(R*self.deltas)-alpha_term
            good = np.where(appmag < v5[i])
            simData['NEODist'][i] = np.max(self.deltas[good])

        simData['NEOX'] = -simData['NEODist']*np.cos(simData[self.elongCol])
        simData['NEOY'] = simData['NEODist']*np.sin(simData[self.elongCol])-1.

        return simData
