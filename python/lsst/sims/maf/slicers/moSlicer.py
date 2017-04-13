import numpy as np
import numpy.ma as ma
import pandas as pd
import warnings

from .baseSlicer import BaseSlicer
from lsst.sims.maf.plots.moPlotters import MetricVsH, MetricVsOrbit

from lsst.sims.movingObjects import Orbits

__all__ = ['MoObjSlicer']


class MoObjSlicer(BaseSlicer):
    """ Slice moving object _observations_, per object and optionally clone/per H value.

    Iteration over the MoObjSlicer will go as:
    - iterate over each orbit;
    - if Hrange is not None, for each orbit, iterate over Hrange.
    """
    def __init__(self, verbose=True, badval=0):
        super(MoObjSlicer, self).__init__(verbose=verbose, badval=badval)
        # Set default plotFuncs.
        self.plotFuncs = [MetricVsH(),
                          MetricVsOrbit(xaxis='q', yaxis='e'),
                          MetricVsOrbit(xaxis='q', yaxis='inc')]

    def readOrbits(self, orbitFile, Hrange, delim=None, skiprows=None):
        # Use sims_movingObjects to read orbit files.
        orb = Orbits()
        orb.readOrbits(orbitFile, delim=delim, skiprows=skiprows)
        self.orbits = orb.orbits
        # Then go on as previously. Need to refactor this into 'setupSlicer' style.
        self.nSso = len(self.orbits)
        self.slicePoints = {}
        self.slicePoints['orbits'] = self.orbits
        # See if we're cloning orbits.
        self.Hrange = Hrange
        # And set the slicer shape/size.
        if self.Hrange is not None:
            self.shape = [self.nSso, len(Hrange)]
            self.slicePoints['H'] = Hrange
        else:
            self.shape = [self.nSso, 1]
            self.slicePoints['H'] = self.orbits['H']
        # Set the rest of the slicePoint information once
        self.nslice = self.shape[0] * self.shape[1]

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

    def __next__(self):
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
        if isinstance(otherSlicer, MoObjSlicer):
            if otherSlicer.obsfile == self.obsfile:
                if np.all(otherSlicer.slicePoints['H'] == self.slicePoints['H']):
                    result = True
        return result

    def writeData(self, outfilename, metricValues, metricName='',
                  simDataName='', constraint=None, metadata='',
                  plotDict=None, displayDict=None):
        """
        Cheap and dirty write to disk.
        Need to expand to include writing summary statistics to disk and info about slicer.
        """
        df = pd.DataFrame(metricValues, columns=self.Hrange, index=None)
        df.to_hdf(outfilename.replace('.npz', '.h5'), 'df_with_missing')

    def readData(self, infilename):
        "Cheap and dirty read."
        slicer = MoObjSlicer()
        df = pd.read_hdf(infilename, 'df_with_missing')
        slicer.Hrange = df.columns.values
        slicer.slicePoints['H'] = slicer.Hrange
        slicer.shape = [len(df.values), len(slicer.Hrange)]
        slicer.orbits = None
        metricValues = ma.MaskedArray(data=df.values,
                                      mask=np.zeros(slicer.shape, 'bool'),
                                      fill_value=slicer.badval)
        try:
            metricValues.mask = np.where(np.isnan(df.values), 1, 0)
        except TypeError:
            warnings.warn('Could not mask metricValues, as they are complex type.')
        return metricValues, slicer
