# Example of more complex metric 
# Takes multiple columns of data (although 'night' could be calculable from 'expmjd')
# Returns variable length array of data
# Uses multiple reduce functions

import numpy as np
from .baseMetric import BaseMetric

class VisitGroupsMetric(BaseMetric):
    """Count the number of visits per night within deltaTmin and deltaTmax."""
    def __init__(self, timesCol='expMJD', nightsCol='night', 
                 deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0, minNVisits=2, window=30, minNNights=3,
                 **kwargs):
        """
        Instantiate metric.
        
        'timesCol' = column with the time of the visit (default expmjd),
        'nightsCol' = column with the night of the visit (default night),
        'deltaTmin' = minimum time of window: units are days (default 15 min),
        'deltaTmax' = maximum time of window: units are days (default 90 min),
        'minNVisits' = the minimum number of visits within a night (with spacing between deltaTmin/max from any other visit) required,
        'window' = the number of nights to consider within a window (for reduce methods),
        'minNNights' = the minimum required number of nights within window to make a full 'group'.
        """
        self.times = timesCol   
        self.nights = nightsCol
        eps = 1e-10
        self.deltaTmin = deltaTmin - eps
        self.deltaTmax = deltaTmax
        self.minNVisits = minNVisits
        self.window = window
        self.minNNights = minNNights
        super(VisitGroupsMetric, self).__init__([self.times, self.nights], **kwargs)

    def run(self, dataSlice):
        """
        Return a dictionary of:
         the number of visits within a night (within delta tmin/tmax of another visit),
         and the nights with visits > minNVisits.
        Count two visits which are within tmin of each other, but which have another visit
         within tmin/tmax interval, as one and a half (instead of two).

        So for example: 4 visits, where 1, 2, 3 were all within deltaTMax of each other, and 4 was later but
          within deltaTmax of visit 3 -- would give you 4 visits. If visit 1 and 2 were closer together
          than deltaTmin, the two would be counted as 1.5 visits together (if only 1 and 2 existed,
          then there would be 0 visits as none would be within the qualifying time interval).
        """
        uniquenights = np.unique(dataSlice[self.nights])
        nights = []
        visitNum = []
        # Find the nights with visits within deltaTmin/max of one another and count the number of visits
        for n in uniquenights:
            condition = (dataSlice[self.nights] == n)
            times = np.sort(dataSlice[self.times][condition])
            nvisits = 0
            ntooclose = 0
            # Calculate difference between each visit and time of previous visit (tnext- tnow)
            timediff = np.diff(times)
            timegood = np.where((timediff <= self.deltaTmax) & (timediff >= self.deltaTmin), True, False)
            timetooclose = np.where(timediff < self.deltaTmin, True, False)
            if len(timegood)>1:
                # Count visits for all but last index in timediff
                for tg1, ttc1, tg2, ttc2 in zip(timegood[:-1], timetooclose[:-1], timegood[1:], timetooclose[1:]):
                    if tg1:
                        nvisits += 1
                        if not tg2:
                            nvisits += 1
                    if ttc1:
                        ntooclose += 1
                        if not tg2 and not ttc2:
                            ntooclose += 1
                # Take care of last timediff
                if timegood[-1]:
                    nvisits += 2 #close out visit sequence
                if timetooclose[-1]:
                    ntooclose += 1
                    if not timegood[-2] and not timetooclose[-2]:
                        ntooclose += 1
            else:
                if timegood:
                    nvisits += 2
            # Count up all visits for night.
            if nvisits > 0:
                nvisits = nvisits + ntooclose/2.0
                visitNum.append(nvisits)
                nights.append(n)
        # Convert to numpy arrays.
        visitNum = np.array(visitNum)
        nights = np.array(nights)
        metricval = {'visits':visitNum, 'nights':nights}
        if len(visitNum) == 0:
            return self.badval
        return metricval
        
    def reduceMedian(self, metricval):
        """Reduce to median number of visits per night (2 visits = 1 pair)."""
        return np.median(metricval['visits'])
        
    def reduceNNightsWithNVisits(self, metricval):
        """Reduce to total number of nights with more than 'minNVisits' visits."""
        condition = (metricval['visits'] >= self.minNVisits)
        return len(metricval['visits'][condition])

    def _inWindow(self, visits, nights, night, window, minNVisits):
        condition = ((nights >= night) & (nights < night+window))
        condition2 = (visits[condition] >= minNVisits)
        return visits[condition][condition2], nights[condition][condition2]
    
    def reduceNVisitsInWindow(self, metricval):
        """Reduce to max number of total visits on all nights with more than minNVisits, within any 'window' (default=30 nights)."""
        maxnvisits = 0
        for n in metricval['nights']:
            vw, nw = self._inWindow(metricval['visits'], metricval['nights'], n, self.window, self.minNVisits)
            maxnvisits = max((vw.sum(), maxnvisits))
        return maxnvisits
    
    def reduceNNightsInWindow(self, metricval):
        """Reduce to max number of nights with more than minNVisits, within 'window' over all windows."""
        maxnights = 0
        for n in metricval['nights']:
            vw, nw = self._inWindow(metricval['visits'], metricval['nights'], n, self.window, self.minNVisits)
            maxnights = max(len(nw), maxnights)
        return maxnights

    def _inLunation(self, visits, nights, lunationStart, lunationLength):
        condition = ((nights >= lunationStart) & (nights < lunationStart+lunationLength))
        return visits[condition], nights[condition]
    
    def reduceNLunations(self, metricval):
        """Reduce to number of lunations (unique 30 day windows) that contain at least one 'group':
        a set of more than minNVisits per night, with more than minNNights of visits within 'window' time period.
        """ 
        lunationLength = 30
        lunations = np.arange(metricval['nights'][0], metricval['nights'][-1]+lunationLength/2.0, lunationLength)
        nLunations = 0
        for l in lunations:
            # Find visits within lunation.
            vl, nl = self._inLunation(metricval['visits'], metricval['nights'], l, lunationLength)
            for n in nl:
                # Find visits which are in groups within the lunation.
                vw, nw = self._inWindow(vl, nl, n, self.window, self.minNVisits)
                if len(nw) >= self.minNNights:
                    nLunations += 1
                    break
        return nLunations

    def reduceMaxSeqLunations(self, metricval):
        """Count the max number of sequential lunations (unique 30 day windows) that contain at least one 'group':
        a set of more than minNVisits per night, with more than minNNights of visits within 'window' time period.
        """
        lunationLength = 30
        lunations = np.arange(metricval['nights'][0], metricval['nights'][-1]+lunationLength/2.0, lunationLength)
        maxSequence = 0
        curSequence = 0
        inSeq = False        
        for l in lunations:
            # Find visits within lunation.
            vl, nl = self._inLunation(metricval['visits'], metricval['nights'], l, lunationLength)
            # If no visits this lunation:
            if len(vl) == 0:
                inSeq = False
                maxSequence = max(maxSequence, curSequence)
                curSequence = 0
            # Else, look to see if groups can be made from the visits.
            for n in nl:
                # Find visits which are in groups within the lunation.
                vw, nw = self._inWindow(vl, nl, n, self.window, self.minNVisits)
                # If there was a group within this lunation:
                if len(nw) >= self.minNNights:
                    curSequence += 1
                    inSeq = True
                    break
                # Otherwise we're not in a sequence (anymore, or still).
                else:
                    inSeq = False
                    maxSequence = max(maxSequence, curSequence)
                    curSequence = 0
        # Pick up last sequence if were in a sequence at last lunation.
        maxSequence = max(maxSequence, curSequence)
        return maxSequence
