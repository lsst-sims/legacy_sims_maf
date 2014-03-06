import numpy as np
from .complexMetrics import ComplexMetric

class CompletenessMetric(ComplexMetric):
    """compute the completeness and joint completeness """
    def __init__(self, plotParams=None,metricName='completeness', u=0., g=0.,r=0.,i=0.,z=0.,y=0.):
        """compute the (joint)completeness for the given filters.  Any filter with a value greater than zero gets computed """
        self.filtercol = 'filter'
        super(CompletenessMetric,self).__init__(self.filtercol,metricName=metricName, plotParams=plotParams)
        self.filters = np.array(['u','g','r','i','z','y'])
        self.cvals = np.array([u,g,r,i,z,y])
        good = np.where(self.cvals > 0)
        self.filters=self.filters[good]
        self.cvals = self.cvals[good]
        
    def run(self,dataSlice):
        """compute the completeness for each filter, and then the minimum completeness for each slice """
        all_complete=[]
        for i,f in enumerate(self.filters):
            completeness = np.size(np.where(dataSlice[self.filtercol] == f)[0])/np.float(self.cvals[i])
            all_complete.append(completeness)
        all_complete.append(np.min(all_complete))
        
        return all_complete
    
    def reduceU(self,completeness):
        return completeness[np.where(self.filters == 'u')[0]]
    def reduceG(self,completeness):
        return completeness[np.where(self.filters == 'g')[0]]
    def reduceR(self,completeness):
        return completeness[np.where(self.filters == 'r')[0]]
    def reduceI(self,completeness):
        return completeness[np.where(self.filters == 'i')[0]]
    def reduceZ(self,completeness):
        return completeness[np.where(self.filters == 'z')[0]]
    def reduceY(self,completeness):
        return completeness[np.where(self.filters == 'y')[0]]
    def reduceJoint(self,completeness):
        """The joint completeness is just the minimum completeness for a point/field"""
        return completeness[-1]
    
    
