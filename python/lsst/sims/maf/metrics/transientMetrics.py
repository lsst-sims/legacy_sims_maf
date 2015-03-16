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
                 nPerLC=1, nFilters=1,
                 **kwargs):
        """
        transDuration = how long the transient lasts (days)
        peakTime = How long it takes to reach the peak magnitude (days)
        riseSlope = slope of the light curve before peak time (mags/day) (XXX -- should be negative since mags are backwards?)
        declineSlope = slope of the light curve after peak time (mags/day)
        (ugrizy)Peak = peak magnitude in each filter
        surveyDuration = length of survey (years)
        surveyStart = MJD for the survey start date (otherwise us the time of the first observation)
        detectM5Plus = an observation will count as a detection if the light curve magnitude is brighter
                       than m5+detectM5Plus
        nPerLC = number of points to light curve for a object to be counted (in a unique filter)
        nFilters = number of filters that need to be observed for an object to be counted
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
        self.nPerLC = nPerLC
        self.nFilters = nFilters

    def run(self, dataSlice, slicePoint=None):

        # XXX--Should I loop this over a few phase-shifts to get a better measure? Maybe only needed in the more complicated transient metrics?

        # Total number of transients that could go off back-to-back
        nTransMax = np.floor(self.surveyDuration/(self.transDuration/365.25))
        if self.surveyStart is None:
            surveyStart = dataSlice[self.mjdCol].min()
        time = (dataSlice[self.mjdCol] - surveyStart) % self.transDuration
        lcMags = np.zeros(dataSlice.size, dtype=float)

        # Which lightcurve does each point belong to
        lcNumber = np.floor((dataSlice[self.mjdCol]-surveyStart)/self.transDuration)

        rise = np.where(time <= self.peakTime)
        lcMags[rise] += self.riseSlope*time[rise]-self.riseSlope*self.peakTime
        decline = np.where(time > self.peakTime)
        lcMags[decline] += self.declineSlope*time[decline]-self.declineSlope*(self.transDuration-self.peakTime)

        for key in self.peaks.keys():
            fMatch = np.where(dataSlice[self.filterCol] == key)
            lcMags[fMatch] += self.peaks[key]

        # How many criteria needs to be passed
        detectThresh = 0

        # flag points that are above the SNR limit
        detected = np.zeros(dataSlice.size, dtype=int)
        detected[np.where(lcMags < dataSlice[self.m5Col] + self.detectM5Plus)] = 1
        detectThresh += 1

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
                #nPhase = np.size(np.unique(phaseSections))
                for filtName in ufilters:
                    good = np.where(dataSlice[self.filterCol][le:ri][points] == filtName)
                    if np.size(np.unique(phaseSections[good])) >= self.nPerLC:
                        detected[le:ri] += 1

        nDetected = np.size(np.unique(lcNumber[np.where(detected >= detectThresh)]))

        return float(nDetected)/nTransMax
