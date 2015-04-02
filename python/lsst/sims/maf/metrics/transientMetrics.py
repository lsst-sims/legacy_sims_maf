import numpy as np
from .baseMetric import BaseMetric


class TransientMetric(BaseMetric):
    """
    Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
    We are assuming simple light curves with no color evolution.
    """
    def __init__(self, metricName='TransientDetectMetric', mjdCol='expMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter',
                 transDuration=10.,peakTime=5., riseSlope=0., declineSlope=0.,
                 surveyDuration=10., surveyStart=None, detectM5Plus=0.,
                 uPeak=20, gPeak=20, rPeak=20, iPeak=20, zPeak=20, yPeak=20,
                 nPrePeak=0, nPerLC=1, nFilters=1, nPhaseCheck = 1,
                 **kwargs):
        """
        transDuration = how long the transient lasts (days)
        peakTime = How long it takes to reach the peak magnitude (days)
        riseSlope = Slope of the light curve before peak time (mags/day).
                    Should be negative since mags are backwards.
        declineSlope = Slope of the light curve after peak time (mags/day).
                       Should be positive since mags are backwards.
        (ugrizy)Peak = Peak magnitude in each filter.
        surveyDuration = Length of survey (years).
        surveyStart = MJD for the survey start date (otherwise us the time of the first observation).
        detectM5Plus = An observation will be used if the light curve magnitude is brighter
                       than m5+detectM5Plus.
        nPrePeak = Number of observations (any filter(s)) to demand before peakTime
                  before saying a transient has been detected.
                  (If one does not trust detection on a single visit)
        nPerLC = Number of sections of the light curve that must be sampled above the detectM5Plus theshold
                 (in a single filter) for the light curve to be counted. For example,
                 setting nPerLC = 2 means a light curve  is only considered detected if there
                 is at least 1 observation in the first half of the LC,
                 and at least one in the second half of the LC.  nPerLC = 4 means each quarter of the light curve
                 must be detected to count.
        nFilters = Number of filters that need to be observed for an object to be counted as detected.
        nPhaseCheck = Sets the number of phases that should be checked.  One can imagine pathological
                      cadences where many objects pass the detection criteria, but would not if the observations
                      were offset by a phase-shift.
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(TransientMetric, self).__init__(col=[self.mjdCol, self.m5Col,self.filterCol],
                                                    units='Fraction Detected',
                                                    metricName=metricName,**kwargs)
        self.peaks = {'u':uPeak,'g':gPeak,'r':rPeak,'i':iPeak,'z':zPeak,'y':yPeak}
        self.transDuration = transDuration
        self.peakTime = peakTime
        self.riseSlope = riseSlope
        self.declineSlope = declineSlope
        self.surveyDuration = surveyDuration
        self.surveyStart = surveyStart
        self.detectM5Plus = detectM5Plus
        self.nPrePeak = nPrePeak
        self.nPerLC = nPerLC
        self.nFilters = nFilters
        self.nPhaseCheck = nPhaseCheck

    def run(self, dataSlice, slicePoint=None):

        # Total number of transients that could go off back-to-back
        nTransMax = np.floor(self.surveyDuration/(self.transDuration/365.25))
        tshifts = np.arange(self.nPhaseCheck)*self.transDuration/float(self.nPhaseCheck)
        nDetected = 0
        nTransMax = 0
        for tshift in tshifts:
            # Compute the total number of back-to-back transients are possible to detect
            # given the survey duration and the transient duration.
            nTransMax += np.floor(self.surveyDuration/(self.transDuration/365.25))
            if tshift != 0:
                nTransMax -= 1
            if self.surveyStart is None:
                surveyStart = dataSlice[self.mjdCol].min()
            time = (dataSlice[self.mjdCol] - surveyStart + tshift) % self.transDuration
            lcMags = np.zeros(dataSlice.size, dtype=float)

            # Which lightcurve does each point belong to
            lcNumber = np.floor((dataSlice[self.mjdCol]-surveyStart)/self.transDuration)

            rise = np.where(time <= self.peakTime)
            lcMags[rise] += self.riseSlope*time[rise]-self.riseSlope*self.peakTime
            decline = np.where(time > self.peakTime)
            lcMags[decline] += self.declineSlope*(time[decline]-self.peakTime)

            for key in self.peaks.keys():
                fMatch = np.where(dataSlice[self.filterCol] == key)
                lcMags[fMatch] += self.peaks[key]

            # How many criteria needs to be passed
            detectThresh = 0

            # Flag points that are above the SNR limit
            detected = np.zeros(dataSlice.size, dtype=int)
            detected[np.where(lcMags < dataSlice[self.m5Col] + self.detectM5Plus)] += 1
            detectThresh += 1

            # If we demand points on the rise
            if self.nPrePeak > 0:
                detectThresh += 1
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                time = time[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')
                # Note here I'm using np.searchsorted to basically do a 'group by'
                # might be clearer to use scipy.ndimage.measurements.find_objects or pandas, but
                # this numpy function is known for being efficient.
                for le,ri in zip(left,right):
                    # Number of points where there are a detection
                    good = np.where(time[le:ri] < self.peakTime)
                    nd = np.sum(detected[le:ri][good])
                    if nd >= self.nPrePeak:
                        detected[le:ri] += 1

            # Check if we need multiple points per light curve or multiple filters
            if (self.nPerLC > 1) | (self.nFilters > 1) :
                # make sure things are sorted by time
                ord = np.argsort(dataSlice[self.mjdCol])
                dataSlice = dataSlice[ord]
                detected = detected[ord]
                lcNumber = lcNumber[ord]
                ulcNumber = np.unique(lcNumber)
                left = np.searchsorted(lcNumber, ulcNumber)
                right = np.searchsorted(lcNumber, ulcNumber, side='right')
                detectThresh += self.nFilters

                for le,ri in zip(left,right):
                    points = np.where(detected[le:ri] > 0)
                    ufilters = np.unique(dataSlice[self.filterCol][le:ri][points])
                    phaseSections = np.floor(time[le:ri][points]/self.transDuration * self.nPerLC)
                    for filtName in ufilters:
                        good = np.where(dataSlice[self.filterCol][le:ri][points] == filtName)
                        if np.size(np.unique(phaseSections[good])) >= self.nPerLC:
                            detected[le:ri] += 1

            # Find the unique number of light curves that passed the required number of conditions
            nDetected += np.size(np.unique(lcNumber[np.where(detected >= detectThresh)]))

        # Rather than keeping a single "detected" variable, maybe make a mask for each criteria, then
        # reduce functions like: reduce_singleDetect, reduce_NDetect, reduce_PerLC, reduce_perFilter.
        # The way I'm running now it would speed things up.

        return float(nDetected)/nTransMax
