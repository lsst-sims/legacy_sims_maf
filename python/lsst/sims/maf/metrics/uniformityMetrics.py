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
        and compare to a uniform cumulative distribution.
        Perfectly uniform observations will score a 0, while pure non-uniformity is 1."""
        # If only one observation, there is no uniformity
        if dataSlice[self.expMJDcol].size == 1:
            return 1
        # Scale dates to lie between 0 and 1, where 0 is the first observation date and 1 is surveyLength
        dates = (dataSlice[self.expMJDcol]-dataSlice[self.expMJDcol].min())/(self.surveyLength*365.25)
        dates.sort() # Just to be sure
        n_cum = np.arange(1,dates.size+1)/float(dates.size) # Cumulative distribution of dates
        D_max = np.max(np.abs(n_cum-dates-dates[1])) # For a uniform distribution, dates = n_cum
        return D_max
        
