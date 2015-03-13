import numpy as np
from .baseMetric import BaseMetric

class TransientDetectMetric(BaseMetric):
    """
    Calculate what fraction of the transients would be detected. Best paired with a spatial slicer.
    We are assuming simple light curves with no color evolution.
    """
    def __init__(self, metricName='TransientDetectMetric', mjdCol='expMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', transDuration=10.,
                 peakTime=5., riseSlope=0., declineSlope=0.,
                 uPeak=20, gPeak=20, rPeak=20, iPeak=20, zPeak=20, yPeak=20,
                 surveyDuration=10., surveyStart=None, detectM5Plus=0., **kwargs):
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
        """
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        super(TransientDetectMetric, self).__init__(col=[self.mjdCol, self.m5Col,self.filterCol],
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

    def run(self, dataSlice, slicePoint=None):
        """

        """

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

        detected = np.zeros(dataSlice.size, dtype=int)
        detected[np.where(lcMags < dataSlice[self.m5Col] + self.detectM5Plus)] = 1

        nDetected = np.size(np.unique(lcNumber[np.where(detected == 1)]))

        return float(nDetected)/nTransMax


# Now, these are going to be fairly similar metrics--it's tempting to make them complex metrics and just have reduce functions for detect, LC, color...


#class TransientLCMetric(BaseMetric):
    #XXX -- similar to detect, but now demand that there are N observations in at least one filter to count

#class TransientColorMetric(BaseMetric):
    #XXX -- similar to above, but now demand the light curve is observed with N well-spaced observations in M different filters.
