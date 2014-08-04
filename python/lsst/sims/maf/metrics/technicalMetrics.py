import numpy as np
from .baseMetric import BaseMetric


class OpenShutterMetric(BaseMetric):
   """Compute the amount of time the shutter is open. """
   def __init__(self, readTime=2., shutterTime=2.,
                metricName='OpenShutterMetric',
                exptimeCol='visitExpTime', **kwargs):
        self.exptimeCol = exptimeCol
        super(OpenShutterMetric,self).__init__(col=self.exptimeCol, metricName=metricName, units='sec')
        self.readTime = readTime
        self.shutterTime = shutterTime
    
   def run(self, dataSlice, slicePoint=None):
       result = np.sum(dataSlice[self.exptimeCol] - self.readTime - self.shutterTime)
       return result

class OpenShutterFractionMetric(BaseMetric):
   """Compute the fraction of time the shutter is open compared to the total time spent observing. """
   def __init__(self, readTime=2., shutterTime=2,
                metricName='OpenShutterFracMetric',
                slewTimeCol='slewTime', exptimeCol='visitExpTime', **kwargs):
       self.exptimeCol = exptimeCol
       self.slewTimeCol = slewTimeCol
       super(OpenShutterFractionMetric,self).__init__(col=[self.exptimeCol, self.slewTimeCol],
                                                  metricName=metricName, units='frac')
       self.units = 'OpenShutter/TotalTime'
       self.readTime = readTime
       self.shutterTime = shutterTime

   def run(self, dataSlice, slicePoint=None):
       result = (np.sum(dataSlice[self.exptimeCol] - self.readTime - self.shutterTime)
                 / np.sum(dataSlice[self.slewTimeCol] + dataSlice[self.exptimeCol]))
       return result


class CompletenessMetric(BaseMetric):
    """Compute the completeness and joint completeness """
    def __init__(self, filterColName='filter', metricName='Completeness',
                 u=0, g=0, r=0, i=0, z=0, y=0, **kwargs):
        """Compute the completeness for the each of the given filters and the
        joint completeness across all filters.
                 
        Completeness calculated in any filter with a requested 'nvisits' value greater than 0, range is 0-1."""
        self.filterCol = filterColName
        super(CompletenessMetric,self).__init__(col=self.filterCol, metricName=metricName, **kwargs)
        self.nvisitsRequested = np.array([u, g, r, i, z, y])
        self.filters = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # Remove filters from consideration where number of visits requested is zero.
        good = np.where(self.nvisitsRequested > 0)        
        self.nvisitsRequested = self.nvisitsRequested[good]
        self.filters = self.filters[good]
        # Raise exception if number of visits wasn't changed from the default, for at least one filter.
        if len(self.filters) == 0:
            raise ValueError('Please set the requested number of visits for at least one filter.')
        # Set an order for the reduce functions (for display purposes only).
        for i, f in enumerate(('u', 'g', 'r', 'i', 'z', 'y', 'Joint')):
            self.reduceOrder[f] = i
        
    def run(self, dataSlice, slicePoint=None):
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
    
    
