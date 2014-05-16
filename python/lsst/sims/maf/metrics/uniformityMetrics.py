import numpy as np
import scipy.stats as stats

from .baseMetric import BaseMetric

class UniformityMetric(BaseMetric):
    """Calculate how uniformly the observations are spaced in time.  Returns a value between -1 and 1.  A value of zero means the observations are perfectly uniform.  """
    def __init__(self, expMJDcol='expMJD', units='',
                 surveyLength=10., **kwargs):
        """surveyLength = time span of survey (years) """
        cols = [expMJDcol]
        super(UniformityMetric,self).__init__(cols, units=units, **kwargs)
        self.expMJDcol = expMJDcol
        self.surveyLength = surveyLength
        self.metricDtype=float


    def run(self,dataSlice):
        """Based on how a KS-Test _actually_ works...
        Look at the cumulative distribution of observations dates,
        and compare to a uniform cumulative distribution.   """
        # If only one observation, there is no uniformity
        if dataSlice[self.expMJDcol].size == 1:
            return 1
        # Scale dates to lie between 0 and 1, where 0 is the first observation date and 1 is surveyLength
        dates = (dataSlice[self.expMJDcol]-dataSlice[self.expMJDcol].min())/(self.surveyLength*365.25)
        dates.sort() # Just to be sure
        n_cum = np.arange(1,dates.size+1)/float(dates.size) # Cumulative distribution of dates
        D_max = np.max(np.abs(n_cum-dates-dates[1])) # For a uniform distribution, dates = n_cum
        return D_max
        
    def Steverun(self, dataSlice):
        # XXX - the algorithm that Steve and Zeljko came up with.  Doesn't do a good job measuring uniformity since it gives very different values for a survey with all the observations on the 1st day compared to one with all the observations on the middle day.  
        self.start=49353.032079
        dates = (dataSlice[self.expMJDcol]-self.start)/(self.surveyLength*365.25)
        obsHist,bins = np.histogram(dates, bins=20,range=[0,1],density=True)
        obsHist = np.cumsum(obsHist/obsHist.sum())
        modelHist = np.arange(0,obsHist.size,1)/float(obsHist.size-1)
        import pdb ; pdb.set_trace()
        f1 = 1./float(obsHist.size)*np.sum(np.abs(obsHist-modelHist))
        good = np.where(obsHist-modelHist != 0) # Elminate divide by zero
        f2 = 1./float(good[0].size)*np.sum((obsHist[good]-modelHist[good])/np.abs(obsHist[good]-modelHist[good]) )
        if np.abs(f2.round) != 1:
            f2 = 1 #eliminate instances where all the observations are at the midpoint results in f2 = 0
        
        return f1*f2
    
