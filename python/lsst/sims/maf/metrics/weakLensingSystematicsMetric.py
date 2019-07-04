import numpy as np
from .metrics import BaseMetric, ExgalM5
from .maps import DustMap
from .slicers import HealpixSlicer
from lsst.sims.utils import angularSeparation


__all__ = ['NumberOfVisitsMetric']

class NumberOfVisitsMetric(BaseMetric):

    
    
    def __init__(self,
                 runName,
                 maps,
                 depthlim,
                 Stacker=stackers.RandomDitherFieldPerVisitStacker(
                                        degrees=True),
                 metricName='AverageVisitsMetric',
                 **kwargs):
        """Weak Lensing systematics metric

        Computes the number of visits per object for a healpix
        grid of points, within the WFD, after LSS cuts"""
        
        super(AverageVisitsMetric, self).__init__(
            metricName=metricName, col=['fieldId', 'fieldDec', 'fiveSigmaDepth'],
            maps=maps, **kwargs
            )
        self.FOVradius = 1.75
        self.runName = runName
        self.Stacker = Stacker
        self.ExgalM5 = ExgalM5()
        self.depthlim = depthlim

    def run(self, dataSlice, slicePoint=None):
        """runs the metric

        Args:
            dataSlice (ndarray): positional data from querying the database
            slicePoint (dict): queried data along with data from stackers
        Returns:
            (int): total number of visits at this healpix point
        """
        result = 0
        if slicePoint['ebv'] > 0.2:
            return self.badval
        ExgalM5 = self.ExgalM5.run(dataSlice=dataSlice, slicePoint=slicePoint)
        if ExgalM5 < self.depthlim:
            return self.badval
        for datum in dataSlice:
            if angularSeparation(
                datum[3],
                datum[4],
                slicePoint['ra']*np.degrees(1),
                slicePoint['dec']*np.degrees(1)
                ) < self.FOVradius:
                result += 1
        return result

