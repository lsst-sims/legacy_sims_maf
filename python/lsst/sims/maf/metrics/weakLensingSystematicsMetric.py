from .baseMetric import BaseMetric
from .exgalM5 import ExgalM5
from ..maps import DustMap


__all__ = ['NumberOfVisitsMetric']

class NumberOfVisitsMetric(BaseMetric):
    """Note:
        Should be run with the HealpixSlicer. If using dithering (which
        should be the case unless dithering is already implemented in the run)
        then should be run with a stacker and appropriate column names for 
        dithered RA and Dec should be provided.
    """

    
    def __init__(self,
                 maps,
                 depthlim=24.5,
                 metricName='NumberOfVisitsMetric',
                 **kwargs):
        """Weak Lensing systematics metric

        Computes the average number of visits per point on a HEALPix grid
        after E(B-V) and co-added depth cuts.
        """
        
        super().__init__(
            metricName=metricName, 
            col=['fieldId', 'fieldDec', 'fiveSigmaDepth'],
            maps=maps, 
            **kwargs
            )
        self.ExgalM5 = ExgalM5()
        self.depthlim = depthlim

    def run(self, dataSlice, slicePoint=None):
        """runs the metric

        Args:
            dataSlice (ndarray): positional data from querying the database
            slicePoint (dict): queried data along with data from stackers
        Returns:
            the number of visits that can observe this healpix point.
        """
        if slicePoint['ebv'] > 0.2:
            return self.badval
        ExgalM5 = self.ExgalM5.run(dataSlice=dataSlice, slicePoint=slicePoint)
        if ExgalM5 < self.depthlim:
            return self.badval
        return len(dataSlice)

