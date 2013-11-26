# Example of more complex metric 
# Takes multiple columns of data (although 'night' could be calculable from 'expmjd')
# Returns variable length array of data
# Uses multiple reduce functions

import numpy as np
from .baseMetric import BaseMetric

class VisitPairsMetric(BaseMetric):
    """Count the number of pairs of visits per night within deltaTmin and deltaTmax."""
    def __init__(self, timesCol='expmjd', nightsCol='night', metricName='dtimes',
                 deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0):
        """Instantiate metric.
        
        'timesCol' = column with the time of the visit (default expmjd),
        'nightsCol' = column with the night of the visit (default night),
        'deltaTmin' = minimum time of window,
        'deltaTmax' = maximum time of window."""
        self.times = timesCol   
        self.nights = nightsCol
        self.deltaTmin = deltaTmin
        self.deltaTmax = deltaTmax
        super(VisitPairsMetric, self).__init__([self.times, self.nights], metricName)
        # Dictionary of reduce functions.
        self.reduceFuncs = {'Median': self.reduceMedian,
                            'Mean': self.reduceMean, 
                            'Rms': self.reduceRms,
                            'NNights':self.reduceNNights}
        return

    def run(self, dataSlice):
        nights = np.unique(dataSlice[self.nights])
        visitPairs = np.zeros(len(nights), 'int')        
        for i, n in enumerate(nights):
            condition = (dataSlice[self.nights] == n)
            times = dataSlice[self.times][condition]
            for t in times:
                dt = times - t
                condition = ((dt >= self.deltaTmin) & (dt <= self.deltaTmax))
                visitPairs[i] += len(dt[condition])
        return visitPairs
        
    def reduceMedian(self, pairs):
        """Reduce to median number of pairs per night."""
        return np.median(pairs)

    def reduceMean(self, pairs):
        """Reduce to mean number of pairs per night."""
        return pairs.mean()
    
    def reduceRms(self, pairs):
        """Reduce to std dev of number of pairs per night."""
        return pairs.std()

    def reduceNNights(self, pairs, nPairs=2):
        """Reduce to number of nights with more than 'nPairs' (default=2) visits."""
        condition = (pairs >= nPairs)
        return len(pairs[condition])
        
