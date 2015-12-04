import numpy as np
from .baseMetric import BaseMetric

__all__ = ['SupernovaMetric', 'TemplateExistsMetric', 'UniformityMetric',
           'RapidRevisitMetric', 'NRevisitsMetric', 'IntraNightGapsMetric',
           'InterNightGapsMetric', 'AveGapMetric']

class SupernovaMetric(BaseMetric):
    """
    Measure how many time series meet a given time and filter distribution requirement.
    """
    def __init__(self, metricName='SupernovaMetric',
                 mjdCol='expMJD', filterCol='filter', m5Col='fiveSigmaDepth',
                 units='', redshift=0.,
                 Tmin = -20., Tmax = 60., Nbetween=7, Nfilt=2, Tless = -5., Nless=1,
                 Tmore = 30., Nmore=1, peakGap=15., snrCut=10., singleDepthLimit=23.,
                 resolution=5., badval=-666,
                 uniqueBlocks=False, **kwargs):
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
        singleDepthLimit = require observations in Nfilt different filters to be this
        deep near the peak.  This is a rough approximation for the Science Book
        requirements for a SNR cut.  Ideally, one would import a time-variable SN SED,
        redshift it, and make filter-keyed dictionary of interpolation objects so the
        magnitude of the SN could be calculated at each observation and then use the m5col
        to compute a SNR.
        resolution = time step (days) to consider when calculating observing windows
        uniqueBlocks = should the code count the number of unique sequences that meet
        the requirements (True), or should all sequences that meet the conditions
        be counted (False).

        The filter centers are shifted to the SN restframe and only observations
        with filters between 300 < lam_rest < 900 nm are included

        In the science book, the metric demands Nfilt observations above a SNR cut.
        Here, we demand Nfilt observations near the peak with a given singleDepthLimt.
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(SupernovaMetric, self).__init__(col=[self.mjdCol, self.m5Col, self.filterCol],
                                              metricName=metricName, units=units, badval=badval,
                                              **kwargs)
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
        self.uniqueBlocks = uniqueBlocks
        self.filterNames = np.array(['u','g','r','i','z','y'])
        # Set rough values for the filter effective wavelengths.
        self.filterWave = np.array([375.,476.,621.,754.,870.,980.])/(1.+self.redshift)
        self.filterNames = self.filterNames[np.where( (self.filterWave > 300.) & (self.filterWave < 900.))[0]]
        self.singleDepthLimit = singleDepthLimit

        # It would make sense to put a dict of interpolation functions here keyed on filter that take time
        #and returns the magnitude of a SN.  So, take a SN SED, redshift it, calc it's mag in each filter.
        #repeat for multiple time steps.

    def run(self, dataSlice, slicePoint=None):
        # Cut down to only include filters in correct wave range.
        goodFilters = np.in1d(dataSlice['filter'],self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return (self.badval, self.badval,self.badval)
        dataSlice.sort(order=self.mjdCol)
        time = dataSlice[self.mjdCol]-dataSlice[self.mjdCol].min()
        # Now days in SN rest frame
        time = time/(1.+ self.redshift)
        # Creat time steps to evaluate at
        finetime = np.arange(0.,np.ceil(np.max(time)),self.resolution)
        #index for each time point
        ind = np.arange(finetime.size)
        #index for each time point + Tmax - Tmin
        right = np.searchsorted( time, finetime+self.Tmax-self.Tmin, side='right')
        left = np.searchsorted(time, finetime, side='left')
        # Demand enough visits in window
        good = np.where( (right - left) > self.Nbetween)[0]
        ind = ind[good]
        right = right[good]
        left = left[good]
        result = 0
        # Record the maximum gap near the peak (in rest-frame days)
        maxGap = []
        # Record the total number of observations in a sequence.
        Nobs = []
        right_side = -1
        for i,index in enumerate(ind):
            if i <= right_side:
                pass
            else:
                visits = dataSlice[left[i]:right[i]]
                t = time[left[i]:right[i]]
                t = t-finetime[index]+self.Tmin

                if np.size(np.where(t < self.Tless)[0]) > self.Nless:
                    if np.size(np.where(t > self.Tmore)[0]) > self.Nmore:
                        if np.size(t) > self.Nbetween:
                            ufilters = np.unique(visits[self.filterCol])
                            if np.size(ufilters) >= self.Nfilt: #XXX need to add snr cut here
                                filtersBrightEnough = 0
                                nearPeak = np.where((t > self.Tless) & (t < self.Tmore))
                                ufilters = np.unique(visits[self.filterCol][nearPeak])
                                for f in ufilters:
                                    if np.max(visits[self.m5Col][nearPeak]
                                              [np.where(visits[self.filterCol][nearPeak] == f)]) \
                                              > self.singleDepthLimit:
                                        filtersBrightEnough += 1
                                if filtersBrightEnough >= self.Nfilt:
                                    if np.size(nearPeak) >= 2:
                                        gaps = t[nearPeak][1:]-np.roll(t[nearPeak],1)[1:]
                                    else:
                                        gaps = self.peakGap+1e6
                                    if np.max(gaps) < self.peakGap:
                                        result += 1
                                        if self.uniqueBlocks:
                                            right_side = right[i]
                                        maxGap.append(np.max(gaps))
                                        Nobs.append(np.size(t))
        maxGap = np.array(maxGap)
        Nobs=np.array(Nobs)
        return {'result':result, 'maxGap':maxGap, 'Nobs':Nobs}

    def reduceMedianMaxGap(self,data):
        """The median maximum gap near the peak of the light curve """
        result = np.median(data['maxGap'])
        if np.isnan(result):
            result = self.badval
        return result
    def reduceNsequences(self,data):
        """The number of sequences that met the requirements """
        return data['result']
    def reduceMedianNobs(self,data):
        """Median number of observations covering the entire light curve """
        result = np.median(data['Nobs'])
        if np.isnan(result):
            result = self.badval
        return result

class TemplateExistsMetric(BaseMetric):
    """
    Calculate the fraction of images with a previous template image of desired quality.
    """
    def __init__(self, seeingCol = 'FWHMgeom', expMJDCol='expMJD',
                 metricName='TemplateExistsMetric', **kwargs):
        """
        seeingCol = column with final seeing value (arcsec)
        expMJDCol = column with exposure MJD.
        """
        cols = [seeingCol, expMJDCol]
        super(TemplateExistsMetric, self).__init__(col=cols, metricName=metricName, units='fraction', **kwargs)
        self.seeingCol = seeingCol
        self.expMJDCol = expMJDCol

    def run(self, dataSlice, slicePoint=None):
        # Check that data is sorted in expMJD order
        dataSlice.sort(order=self.expMJDCol)
        # Find the minimum seeing up to a given time
        seeing_mins = np.minimum.accumulate(dataSlice[self.seeingCol])
        # Find the difference between the seeing and the minimum seeing at the previous visit
        seeing_diff = dataSlice[self.seeingCol] - np.roll(seeing_mins,1)
        # First image never has a template; check how many others do
        good = np.where(seeing_diff[1:] >= 0.)[0]
        frac = (good.size)/float(dataSlice[self.seeingCol].size)
        return frac

class UniformityMetric(BaseMetric):
    """
    Calculate how uniformly the observations are spaced in time.  Returns a value between -1 and 1.
    A value of zero means the observations are perfectly uniform.
    """
    def __init__(self, expMJDCol='expMJD', units='',
                 surveyLength=10., **kwargs):
        """surveyLength = time span of survey (years) """
        self.expMJDCol = expMJDCol
        super(UniformityMetric,self).__init__(col=self.expMJDCol, units=units, **kwargs)
        self.surveyLength = surveyLength

    def run(self,dataSlice, slicePoint=None):
        """Based on how a KS-Test works:
        Look at the cumulative distribution of observations dates,
        and compare to a uniform cumulative distribution.
        Perfectly uniform observations will score a 0, while pure non-uniformity is 1."""
        # If only one observation, there is no uniformity
        if dataSlice[self.expMJDCol].size == 1:
            return 1
        # Scale dates to lie between 0 and 1, where 0 is the first observation date and 1 is surveyLength
        dates = (dataSlice[self.expMJDCol]-dataSlice[self.expMJDCol].min())/(self.surveyLength*365.25)
        dates.sort() # Just to be sure
        n_cum = np.arange(1,dates.size+1)/float(dates.size)
        D_max = np.max(np.abs(n_cum-dates-dates[1]))
        return D_max



class RapidRevisitMetric(BaseMetric):
    """
    Calculate uniformity of time between consecutive visits on short timescales (for RAV1).
    """
    def __init__(self, timeCol='expMJD', minNvisits=100,
                 dTmin=40.0/60.0/60.0/24.0, dTmax=30.0/60.0/24.0,
                 metricName='RapidRevisit', **kwargs):
        """
        timeCol = times of visits
        minNvisits = minimum number of visits within dtime in interval
        dTmin = minimum dtime to consider (default 40 seconds)
        dTmax = maximum dtime to consider (default 30 minutes)
        """
        self.timeCol = timeCol
        self.minNvisits = minNvisits
        self.dTmin = dTmin
        self.dTmax = dTmax
        super(RapidRevisitMetric, self).__init__(col=self.timeCol, metricName=metricName, **kwargs)
        # Update minNvisits, as 0 visits will crash algorithm and 1 is nonuniform by definition.
        if self.minNvisits <= 1:
            self.minNvisits = 2

    def run(self, dataSlice, slicePoint=None):
        # Calculate consecutive visit time intervals
        dtimes = np.diff(np.sort(dataSlice[self.timeCol]))
        # Identify dtimes within interval from dTmin/dTmax.
        good = np.where((dtimes >= self.dTmin) & (dtimes <= self.dTmax))[0]
        # If there are not enough visits in this time range, return bad value.
        if good.size < self.minNvisits:
            return self.badval
        # Throw out dtimes outside desired range, and sort, then scale to 0-1.
        dtimes = np.sort(dtimes[good])
        dtimes = (dtimes-dtimes.min()) / float(self.dTmax-self.dTmin)
        # Set up a uniform distribution between 0-1 (to match dtimes).
        uniform_dtimes = np.arange(1, dtimes.size+1, 1)/float(dtimes.size)
        # Look at the differences between our times and the uniform times.
        dmax = np.max(np.abs(uniform_dtimes-dtimes-dtimes[1]))
        return dmax

class NRevisitsMetric(BaseMetric):
    """
    Calculate the number of (consecutive) visits with time differences less than dT.
    """
    def __init__(self, timeCol='expMJD', dT=30.0, normed=False, metricName=None,**kwargs):
        """
        dT = time interval to consider (in minutes, default 30).
        normed = False - returns number of visits
               = True - returns fraction of overall visits
        """
        units = None
        if metricName is None:
            if normed:
                metricName = 'Fraction of revisits faster than %.1f minutes' %(dT)
            else:
                metricName = 'Number of revisits faster than %.1f minutes' %(dT)
                units = '#'
        self.timeCol = timeCol
        self.dT = dT / 60./24. # convert to days
        self.normed = normed
        super(NRevisitsMetric, self).__init__(col=self.timeCol, units=units, metricName=metricName, **kwargs)
        self.metricDtype = 'int'

    def run(self, dataSlice, slicePoint=None):
        dtimes = np.diff(np.sort(dataSlice[self.timeCol]))
        nFastRevisits = np.size(np.where(dtimes<=self.dT)[0])
        if self.normed:
            nFastRevisits = nFastRevisits / float(np.size(dataSlice[self.timeCol]))
        return nFastRevisits

class IntraNightGapsMetric(BaseMetric):
    """
    Calculate the gap between consecutive observations within a night.
    """

    def __init__(self, timeCol='expMJD', nightCol='night', reduceFunc=np.median,
                 metricName='Median Intra-Night Gap',**kwargs):
        """

        """
        units = 'hours'
        self.timeCol = timeCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super(IntraNightGapsMetric,self).__init__(col=[self.timeCol,self.nightCol] ,
                                                  units=units, metricName=metricName, **kwargs)

    def run(self,dataSlice, slicePoint=None):
        dataSlice.sort(order=self.timeCol)
        dt = np.diff(dataSlice[self.timeCol])
        dn = np.diff(dataSlice[self.nightCol])

        good = np.where(dn == 0)
        if np.size(good[0]) == 0:
            result = self.badval
        else:
            result = self.reduceFunc(dt[good])*24
        return result


class InterNightGapsMetric(BaseMetric):
    """
    Calculate the gap between consecutive observations between nights.
    """
    def __init__(self, timeCol='expMJD', nightCol='night', reduceFunc=np.median,
                 metricName='Median Inter-Night Gap',**kwargs):
        """

        """
        units = 'days'
        self.timeCol = timeCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super(InterNightGapsMetric,self).__init__(col=[self.timeCol,self.nightCol] ,
                                                  units=units, metricName=metricName, **kwargs)

    def run(self,dataSlice, slicePoint=None):
        dataSlice.sort(order=self.timeCol)
        unights = np.unique(dataSlice[self.nightCol])
        if np.size(unights) < 2:
            result = self.badval
        else:
            # Find the first and last observation of each night
            firstOfNight = np.searchsorted(dataSlice[self.nightCol], unights)
            lastOfNight = np.searchsorted(dataSlice[self.nightCol], unights, side='right')-1
            diff = dataSlice[self.timeCol][firstOfNight[1:]] - dataSlice[self.timeCol][lastOfNight[:-1]]
            result = self.reduceFunc(diff)
        return result


class AveGapMetric(BaseMetric):
    """
    Calculate the gap between consecutive observations.
    """
    def __init__(self, timeCol='expMJD', nightCol='night', reduceFunc=np.median,
                 metricName='AveGap',**kwargs):
        """

        """
        units = 'hours'
        self.timeCol = timeCol
        self.nightCol = nightCol
        self.reduceFunc = reduceFunc
        super(AveGapMetric,self).__init__(col=[self.timeCol,self.nightCol] ,
                                          units=units, metricName=metricName, **kwargs)

    def run(self,dataSlice, slicePoint=None):

        dataSlice.sort(order=self.timeCol)
        diff = np.diff(dataSlice[self.timeCol])
        result = self.reduceFunc(diff)*24.
        return result
