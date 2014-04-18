# Base class for complex metrics (containing reduce methods). 
# Adds dictionary containing reduce methods.

import inspect
from .baseMetric import BaseMetric

class ComplexMetric(BaseMetric):
    """Base class for complex metrics containing reduce methods."""
    def __init__(self, cols, *args, **kwargs):
        """Instantiate complex metric."""
        super(ComplexMetric, self).__init__(cols, *args, **kwargs)
        # Create dictionary of reduce functions.
        self.reduceFuncs = {}
        for r in inspect.getmembers(self, predicate=inspect.ismethod):
            if r[0].startswith('reduce'):
                reducename = r[0].replace('reduce', '', 1)
                self.reduceFuncs[reducename] = r[1]
        return
    
    def run(self, dataSlice):
        raise NotImplementedError()

    
