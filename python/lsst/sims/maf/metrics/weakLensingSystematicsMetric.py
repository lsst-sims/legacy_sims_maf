from .baseMetric import BaseMetric
from .exgalM5 import ExgalM5
from ..maps import DustMap


__all__ = ['WeakLensingNvisits']

class WeakLensingNvisits(BaseMetric):
    """A proxy metric for WL systematics. Higher value indicated better 
    systematics mitigation.
    
    Note:
        Should be run with the HealpixSlicer. If using dithering (which
        should be the case unless dithering is already implemented in the run)
        then should be run with a stacker and appropriate column names for 
        dithered RA and Dec should be provided.
    """

    
    def __init__(self,
                 maps,
                 depthlim=24.5,
                 ebvlim=0.2,
                 metricName='WeakLensingNvisits',
                 **kwargs):
        """Weak Lensing systematics metric

        Computes the average number of visits per point on a HEALPix grid
        after a maximum E(B-V) cut and a minimum co-added depth cut.
        """
        
        super().__init__(
            metricName=metricName, 
            col=['fiveSigmaDepth'],
            maps=maps, 
            **kwargs
            )
        self.ExgalM5 = ExgalM5()
        self.depthlim = depthlim
        self.ebvlim = ebvlim

    def run(self, dataSlice, slicePoint=None):
        """runs the metric

        Args:
            dataSlice (ndarray): positional data from querying the database
            slicePoint (dict): queried data along with data from stackers
        Returns:
            the number of visits that can observe this healpix point.
        """
        if slicePoint['ebv'] > self.ebvlim:
            return self.badval
        ExgalM5 = self.ExgalM5.run(dataSlice=dataSlice, slicePoint=slicePoint)
        if ExgalM5 < self.depthlim:
            return self.badval
        return len(dataSlice)

