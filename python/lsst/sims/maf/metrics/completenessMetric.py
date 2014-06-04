import numpy as np
from .baseMetric import BaseMetric

class CompletenessMetric(BaseMetric):
    """Compute the completeness and joint completeness """
    def __init__(self, filterColName='filter', metricName='Completeness', u=0, g=0, r=0, i=0, z=0, y=0, **kwargs):
        """Compute the completeness for the each of the given filters and the joint completeness across all filters.
        Completeness calculated in any filter with a requested 'nvisits' value greater than 0."""
        self.filterCol = filterColName
        super(CompletenessMetric,self).__init__(self.filterCol, metricName=metricName, **kwargs)
        self.nvisitsRequested = np.array([u, g, r, i, z, y])
        self.filters = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # Remove filters from consideration where number of visits requested is zero.
        good = np.where(self.nvisitsRequested > 0)        
        self.nvisitsRequested = self.nvisitsRequested[good]
        self.filters = self.filters[good]
        # Raise exception if number of visits wasn't changed from the default, for at least one filter.
        if len(self.filters) == 0:
            raise ValueError('Please set the requested number of visits for at least one filter.')
        
    def run(self, dataSlice):
        """Compute the completeness for each filter, and then the minimum (joint) completeness for each slice."""
        allCompleteness = []
        for f, nVis in zip(self.filters, self.nvisitsRequested):
            filterVisits = np.size(np.where(dataSlice[self.filterCol] == f)[0])
            allCompleteness.append(filterVisits/np.float(nVis))
        allCompleteness.append(np.min(np.array(allCompleteness)))
        return np.array(allCompleteness)
    
    def reduceu(self, completeness):
        if 'u' in self.filters:
            return completeness[np.where(self.filters == 'u')[0]]
        else:
            return 1

    def reduceg(self, completeness):
        if 'g' in self.filters:
            return completeness[np.where(self.filters == 'g')[0]]
        else:
            return 1

    def reducer(self, completeness):
        if 'r' in self.filters:
            return completeness[np.where(self.filters == 'r')[0]]
        else:
            return 1

    def reducei(self, completeness):
        if 'i' in self.filters:
            return completeness[np.where(self.filters == 'i')[0]]
        else:
            return 1

    def reducez(self, completeness):
        if 'z' in self.filters:
            return completeness[np.where(self.filters == 'z')[0]]
        else:
            return 1

    def reducey(self, completeness):
        if 'y' in self.filters:            
            return completeness[np.where(self.filters == 'y')[0]]
        else: 
            return 1

    def reduceJoint(self, completeness):
        """The joint completeness is just the minimum completeness for a point/field"""
        return completeness[-1]
    
    
