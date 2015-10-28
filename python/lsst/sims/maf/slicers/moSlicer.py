import os
import numpy as np
import pandas as pd

from lsst.sims.maf.plots.moPlotters import MetricVsH, MetricVsOrbit
from lsst.sims.maf.objUtils import MoOrbits

__all__ = ['MoSlicer']


class MoSlicer(MoOrbits):

    def __init__(self, orbitfile, Hrange=None):
        """
        Instantiate the MoSlicer object.

        orbitfile = the file with the orbit information on the objects.

        Iteration over the MoSlicer will go as:
          - iterate over each orbit;
            - if Hrange is not None, for each orbit, iterate over Hrange.
        """
        self.slicerName = 'MoSlicer'
        # Read orbits (inherited from MoOrbits).
        self.readOrbits(orbitfile)
        self.slicePoints = {}
        self.slicePoints['orbits'] = self.orbits
        # See if we're cloning orbits.
        self.Hrange = Hrange
        # And set the slicer shape/size.
        if self.Hrange is not None:
            self.slicerShape = [self.nSso, len(Hrange)]
            self.slicePoints['H'] = Hrange
        else:
            self.slicerShape = [self.nSso, 1]
            self.slicePoints['H'] = self.orbits['H']
        # Set the rest of the slicePoint information once
        self.badval = 0
        # Set default plotFuncs.
        self.plotFuncs = [MetricVsH(),
                          MetricVsOrbit(xaxis='q', yaxis='e'),
                          MetricVsOrbit(xaxis='q', yaxis='inc')]


    def readObs(self, obsfile):
        """
        Read observations created by moObs.
        """
        # For now, just read all the observations (should be able to chunk this though).
        self.obsfile = obsfile
        self.allObs = pd.read_table(obsfile, delim_whitespace=True)
        # We may have to rename the first column from '#objId' to 'objId'.
        if self.allObs.columns.values[0].startswith('#'):
            newcols = self.allObs.columns.values
            newcols[0] = newcols[0].replace('#', '')
            self.allObs.columns = newcols
        if 'magFilter' not in self.allObs.columns.values:
            self.allObs['magFilter'] = self.allObs['magV'] + self.allObs['dmagColor']
        if 'velocity' not in self.allObs.columns.values:
            self.allObs['velocity'] = np.sqrt(self.allObs['dradt']**2 + self.allObs['ddecdt']**2)
        if 'visitExpTime' not in self.allObs.columns.values:
            self.allObs['visitExpTime'] = np.zeros(len(self.allObs['objId']), float) + 30.0
        # If we created intermediate data products by pandas, we may have an inadvertent 'index'
        #  column. Since this creates problems later, drop it here.
        if 'index' in self.allObs.columns.values:
            self.allObs.drop('index', axis=1, inplace=True)
        self.subsetObs()

    def subsetObs(self, pandasConstraint=None):
        """
        Choose a subset of all the observations, such as those in a particular time period.
        """
        if pandasConstraint is None:
            self.obs = self.allObs
        else:
            self.obs = self.allObs.query(pandasConstraint)

    def _sliceObs(self, idx):
        """
        Return the observations of ssoId.
        For now this works for any ssoId; in the future, this might only work as ssoId is
         progressively iterated through the series of ssoIds (so we can 'chunk' the reading).
        """
        # Find the matching orbit.
        orb = self.orbits.iloc[idx]
        # Find the matching observations.
        if self.obs['objId'].dtype == 'object':
            obs = self.obs.query('objId == "%s"' %(orb['objId']))
        else:
            obs = self.obs.query('objId == %d' %(orb['objId']))
        # Return the values for H to consider for metric.
        if self.Hrange is not None:
            Hvals = self.Hrange
        else:
            Hvals = np.array([orb['H']], float)
        # Note that ssoObs / obs is a recarray not Dataframe!
        return {'obs': obs.to_records(),
                'orbit': orb,
                'Hvals': Hvals}

    def __iter__(self):
        """
        Iterate through each of the ssoIds.
        """
        self.idx = 0
        return self

    def next(self):
        """
        Returns result of self._getObs when iterating over moSlicer.
        """
        if self.idx >= self.nSso:
            raise StopIteration
        idx = self.idx
        self.idx += 1
        return self._sliceObs(idx)

    def __getitem__(self, idx):
        # This may not be guaranteed to work if/when we implement chunking of the obsfile.
        return self._sliceObs(idx)

    def __eq__(self, otherSlicer):
        """
        Evaluate if two slicers are equal.
        """
        result = False
        if isinstance(otherSlicer, MoSlicer):
            if otherSlicer.obsfile == self.obsfile:
                if np.all(otherSlicer.slicePoints['H'] == self.slicePoints['H']):
                    result = True
        return result

    def __ne__(self, otherSlicer):
        """
        Evaluate if two slicers are not equal.
        """
        if self == otherSlicer:
            return False
        else:
            return True

    def write(self, filename, metricBundle):
        """
        Cheap and dirty write to disk.
        """
        #store = pd.HDFStore(filename+'.h5')
        df = pd.DataFrame(metricBundle.metricValues)
        df.to_csv(filename)
