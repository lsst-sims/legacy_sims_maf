import numpy as np
from .baseMetric import BaseMetric

class SummaryStatsMetric(BaseMetric):
    """ Calculate how efficent the survey is at keeping the shutter open"""
    def __init__(self, metricName='observeEfficency', visitTime=34., visitExpTime=30., visitSlewTime=2., plotParams=None, units=''):
        """ visitTime = time telescope is stationary durring a visit
            visitExpTime = total time shutter is open
        """
        # I assume Opsim slewtime includes settle time?!?!
        # Need to double check that the filter change time is charged properly  
        self.col = ['slewTime', 'expMJD', 'filter', 'night']
        self.visitTime = visitTime
        self.visitExpTime = visitExpTime
        super(SummaryStatsMetric,self).__init__(self.col, metricName, plotParams=plotParams)
    def run(self, dataSlice):
        slewTime = np.sum(dataSlice[self.col[0]])
        expTime = self.visitExpTime*len(dataSlice)
        dataSlice.sort(order='expMJD')
        filter_changes = np.where(dataSlice['filter'] != np.roll(dataSlice['filter'],1))
        time_steps = (dataSlice['expMJD'] - np.roll(dataSlice['expMJD'],1))*24.*60. #in min
        real_changes = np.where((time_steps[filter_changes] > 0) & (time_steps[filter_changes] < 3.5))
        filterTime = np.sum(time_steps[filter_changes][real_changes]  )/24./60. 
        readAndShutter = (self.visitTime - self.visitExpTime)*len(dataSlice)
        totalTime = slewTime + expTime + readAndShutter + filterTime
        print 'Total number of exposures = ', len(dataSlice)
        print 'Total observable time = %.1f hr'%(totalTime/3600.)
        print 'Open Shutter Fraction = ', expTime/totalTime
        print 'Number of filter changes = ', np.size(real_changes)
        print 'Number of filter changes (incl start of night) =', np.size(filter_changes)
        print 'Number of nigths w/observations = ', np.size(np.unique(dataSlice['night']))
        return np.array([expTime/totalTime])
    
    
