# Class for computing the f_0 metric.  Nearly identical 
# to HealpixBinner, but with an added plotting method

import numpy as np
import matplotlib.pyplot as plt
from .healpixBinner import HealpixBinner

class f0Binner(HealpixBinner):
    """f0 spatial binner"""

    def plotF0(self, metricValue, title=None, xlabel='Number of Visits', ylabel='Area (1000s of square degrees)', fignum=None, scale=None, Asky=18000., Nvisit=825):
        """ 
        Note that Asky and Nvisit need to be set for both the binner and the summary statistic for the plot and returned summary stat values to be consistent!"""
        if scale is None:
            scale = (hp.nside2pixarea(self.nside, degrees=True)  / 1000.0)
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        # Expect metricValue to be something like number of visits
        metricValue.sort()
        cumulativeArea = np.arange(1,metricValue.size+1)[::-1]*scale
        plt.plot(metricValue, cumulativeArea,'k-')
        f0Area = summaryStats.f0Area(None,Asky=Asky).run(metricValue)
        f0Nv = summaryStats.f0Nv(None,Nvisit=Nvisit).run(metricValue)
        plt.axvline(x=Nvisits, linewidth=3, color='b')
        plt.axhline(y=Asky/1000., linewidth=3,color='r')
        
        plt.axvline(x=f0Nv, linewidth=3, color='b', 
                    alpha=.5, label=r'f$_0$ Nvisits=%i'%f0Nv)
        plt.axhline(y=f0Area/1000., linewidth=3,color='r', 
                    alpha=.5, label='f$_0$ Area=%f'%F0Area)
        plt.legend(loc='lower left')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)
        
        return fig.number
