import numpy as np
from .complexMetrics import ComplexMetric

class SupernovaMetric(ComplexMetric):
    """Measure how many time serries meet a given time and filter distribution requirement """
    def __init__(self, metricName='SupernovaMetric', mjdcol='expMJD', filtercol='filter',
                 m5col='fivesigma_modified', units='', redshift=0.,
                 Tmin = -20., Tmax = 60., Nbetween=7, Nfilt=2, Tless = -5., Nless=1,
                 Tmore = 30., Nmore=1, peakGap=15., snrCut=10., resolution=1., badval=-666,**kwargs):
        """
        redshift = redshift of the SN.  Used to scale observing dates to SN restframe.
        Tmin = the minimum day to consider the SN.  
        Tmax = the maximum to consider.
        Nbetween = the number of observations to demand between Tmin and Tmax
        Nfilt = number of unique filters that must observe the SN above the snrCut
        Tless = minimum time to consider 'near peak'
        Tmore = max time to consider 'near peak'
        Nless = number of observations to demand before Tless
        Nmore = number of observations to demand after Tmore
        peakGap = maximum gap alowed between observations in the 'near peak' time
        snrCut = require snr above this limit when counting Nfilt XXX-not yet implemented
        resolution = time step (days) to consider when calculating observing windows"""
        
        cols=[mjdcol,filtercol,m5col]
        self.mjdcol = mjdcol
        self.filtercol = filtercol
        super(SupernovaMetric, self).__init__(cols,metricName, units=units,**kwargs)
        self.metricDtype = 'object'
        self.units = units
        self.redshift = redshift
        self.Tmin = Tmin
        self.Tmax = Tmax
        self.Nbetween = Nbetween
        self.Nfilt = Nfilt
        self.Tless = Tless
        self.Nless = Nless
        self.Tmore = Tmore
        self.Nmore = Nmore
        self.peakGap = peakGap
        self.snrCut = snrCut
        self.resolution = resolution
        self.filterNames = np.array(['u','g','r','i','z','y'])
        self.filterWave = np.array([375.,476.,621.,754.,870.,980.])/(1.+self.redshift) # XXX - rough values
        self.filterNames = self.filterNames[np.where( (self.filterWave > 300.) & (self.filterWave < 900.))[0]] #XXX make wave limits kwargs?
        #self.filters = {'u':375.,'g':476.,'r':621.,'i':754.,'z':870.,'y':980.} # XXX-rough placeholder.  Could upgrade to pull from sims throughputs?
        
        self.badval = badval

        # It would make sense to put a dict of interpolation functions here keyed on filter that take time and returns the magnitude of a SN.  So, take a SN SED, redshift it, calc it's mag in each filter.  repeat for multiple time steps.  
        
    def run(self, dataSlice):
        """ """
        # Cut down to only include filters in correct wave range.
        goodFilters = np.in1d(dataSlice['filter'],self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return (self.badval, self.badval,self.badval)
        dataSlice.sort(order=self.mjdcol)
        time = dataSlice[self.mjdcol]-dataSlice[self.mjdcol].min()
        time = time/(1.+ self.redshift) # Now days in SN rest frame
        finetime = np.arange(0.,np.ceil(np.max(time)),self.resolution) # Creat time steps to evaluate at

        # Maybe rather than using a fine time sampling, we should use course sampling and then curves do not need to be distinct?  It would be like saying, a SN goes off every 5 days--how many of those do we sample well?  Maybe even return a fraction as one of the reduce functions to say what fraction of the time is well-sampled.  Maybe even just put in a demandUnique kwarg and let the user decide.  Put in doc that the suggested way to run is resolution=1, demandUnique=True or resolution=5, demandUnique=False.
        
        ind = np.arange(finetime.size) #index for each time point
        right = np.searchsorted(time, finetime+self.Tmax-self.Tmin, side='right') #index for each time point + Tmax
        good = np.where( (right - ind) > self.Nbetween)[0] # Demand enough visits in window
        ind = ind[good]
        right = right[good]
                
        result = 0
        maxGap = [] # Record the maximum gap near the peak (in rest-frame days)
        Nobs = [] # Record the total number of observations in a sequence.
        
        right_side = -1
        for i,index in enumerate(ind):
            if i <= right_side:
                pass
            else:
                inWindow = np.where((time >=  finetime[index:right[i]].min()) & (time <=  finetime[index:right[i]].max()))
                visits = dataSlice[inWindow]
                t = time[inWindow]
                t = t-finetime[index]+self.Tmin
                
                if np.size(np.where(t < self.Tless)[0]) > self.Nless:
                    if np.size(np.where(t > self.Tmore)[0]) > self.Nmore:
                        if np.size(t) > self.Nbetween:
                            if np.size(np.unique(dataSlice[self.filtercol])) > self.Nfilt: #XXX need to add snr cut here
                                result += 1
                                right_side = right[i]
                                nearPeak = t[np.where((t > self.Tless) & (t < self.Tmore))]
                                gaps = nearPeak[1:]-np.roll(nearPeak,1)[1:]
                                maxGap.append(np.max(gaps))
                                Nobs.append(np.size(t))
        maxGap = np.array(maxGap)
        Nobs=np.array(Nobs)
        return (result, maxGap, Nobs)

    def reduceMedianMaxGap(self, (result, maxGap, Nobs)):
        result = np.median(maxGap)
        if np.isnan(result):
            result = self.badval
        return result
    def reduceNsequences(self,(result, maxGap, Nobs)):
        return result
    def reduceMedianNobs(self,(result, maxGap, Nobs)):
        result = np.median(Nobs)
        if np.isnan(result):
            result = self.badval
        return result
    
                                
    
            
        
        
        
