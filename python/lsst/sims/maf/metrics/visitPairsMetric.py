# Example of more complex metric 
# Takes multiple columns of data (although 'night' could be calculable from 'expmjd')
# Returns variable length array of data
# Uses multiple reduce functions

import numpy as np
from .complexMetrics import ComplexMetric

class VisitGroupsMetric(ComplexMetric):
    """Count the number of visits per night within deltaTmin and deltaTmax."""
    def __init__(self, timesCol='expMJD', nightsCol='night', 
                 deltaTmin=15.0/60.0/24.0, deltaTmax=90.0/60.0/24.0, minNVisits=2, window=30, minNNights=3,
                 **kwargs):
        """Instantiate metric.
        
        'timesCol' = column with the time of the visit (default expmjd),
        'nightsCol' = column with the night of the visit (default night),
        'deltaTmin' = minimum time of window (default 15 min),
        'deltaTmax' = maximum time of window (default 90 min),
        'minNVisits' = the minimum number of visits within a night (with spacing between deltaTmin/max from any other visit) required,
        'window' = the number of nights to consider within a window (for reduce methods),
        'minNNights' = the minimum required number of nights within window to make a full 'group'."""
        self.times = timesCol   
        self.nights = nightsCol
        eps = 1e-10
        self.deltaTmin = deltaTmin - eps
        self.deltaTmax = deltaTmax
        self.minNVisits = nVisits
        self.window = window
        self.minNNights = minNNights
        super(VisitGroupsMetric, self).__init__([self.times, self.nights], **kwargs)

    def run(self, dataSlice):
        """Return the number of visits within a night (within delta tmin/tmax of another visit) and the nights with visits > minNVisits."""
        # So for example: 4 visits, where 1, 2, 3 were all within deltaTMax of each other, and 4 was later but within deltaTmax of visit 3 -- would give you 4 visits. 
        # Identify the nights with any visits.
        uniquenights = np.unique(dataSlice[self.nights])
        nights = []
        visitNum = []
        # Identify the nights with sets of visits within time window.
        for i, n in enumerate(uniquenights):
            condition = (dataSlice[self.nights] == n)
            times = dataSlice[self.times][condition]
            nvisits = 0
            for t in times:
                dt = times - t
                condition2 = ((dt >= self.deltaTmin) & (dt <= self.deltaTmax))
                nvisits += len(dt[condition2])
            if nvisits > 0:
                visitNum.append(nvisits)
                nights.append(n)
        # Convert to numpy arrays.
        visitNum = np.array(visitNum)
        nights = np.array(nights)
        if len(visitNum) == 0:
            return self.badval
        return (visitNum, nights)
        
    def reduceMedian(self, (visits, nights)):
        """Reduce to median number of visits per night (2 visits = 1 pair)."""
        return np.median(visits)
    
    '''
    def reduceMean(self, (visits, nights)):
        """Reduce to mean number of visits per night (2 visits = 1 pair)."""
        return visits.mean()

    def reduceRms(self, (visits, nights)):
        """Reduce to std dev of number of visits per night (2 visits = 1 pair)."""
        return visits.std()
    '''
    
    def reduceNNightsWithNVisits(self, (visits, nights), minNVisits=None):
        """Reduce to total number of nights with more than 'minNVisits' visits."""
        if minNVisits==None:
            minNVisits = self.minNVisits
        condition = (visits >= minNVisits)
        return len(visits[condition])

    def reduceNVisitsInWindow(self, (visits, nights), minNVisits=None, window=None):
        """Reduce to max number of total visits on all nights with more than minNVisits, within any 'window' (default=30 nights)."""
        if minNVisits==None:
            minNVisits = self.minNVisits
        if window==None:
            window = self.window
        maxnvisits = 0
        for n in nights:
            condition = ((nights >= n) & (nights <= n+window))
            condition2 = (visits[condition] >= minNVisits)
            maxnvisits = max((visits[condition][condition2].sum(), maxnvisits))
        return maxnvisits
    
    def reduceNNightsInWindow(self, (visits, nights), minNVisits=None, window=None):
        """Reduce to max number of nights with more than minNVisits, within 'window' over all windows."""
        if minNVisits == None:
            minNVisits = self.minNVisits
        if window == None:
            window=self.window
        maxnights = 0
        for n in nights:
            condition = ((nights >=n) & (nights<=n+window))
            condition2 = (visits[condition] >= minNVisits)
            maxnights = max(len(nights[condition][condition2]), maxnights)
        return maxnights

    def reduceNLunations(self, (visits, nights), minNVisits=None, window=None, minNNights=None):
        """Reduce to number of lunations (unique 30 day windows) that contain at least one 'group' - sets of more than minNVisits per night and with more than minNNights nights of observations within 'window' time period."""        
        if minNVisits == None:
            minNVisits = self.minNVisits
        if window == None:
            window=self.window
        if minNNights == None:
            minNNights = self.minNNights
        lunationLength = 30
        lunations = np.arange(nights[0], nights[-1], lunationLength)
        nGroupLunation = 0
        for l in lunations:
            # Identify the visits/nights which are contained within this lunation.
            condition = ((nights >= l) & (nights <= lunationLength))
            for n in nights[condition]:
                # Identify visits/nights within this lunation which are also within window
                condition2 = ((nights[condition] >= n) & (nights[condition] <= n+window))
                # Identify visit/nights with more than minNVisits within group 
                condition3 = (visits[condition][condition2] > minNvisits)
                if len(nights[condition][condition2][condition3]) > minNNights:
                    nGroupLunation += 1
        return nGroupLunation
