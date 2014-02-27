import numpy as np
from .baseMetric import BaseMetric
import ephem

class HourglassMetric(BaseMetric):
    """Plot the filters used as a function of time """
    
    def __init__(self, metricName='hourglass', lat='-30:14:40.7', lon='-70:44:57.9'):

        filtercol = "filter"
        mjdcol = "expMJD"
        nightcol = "night"
        cols = [filtercol, mjdcol, nightcol]
        super(BaseMetric,self).__init__(cols,metricName, units=units)
        self.nightcol = nightcol
        self.mjdcol = mjdcol
        self.filtercol = filtercol
        self.lat = lat
        self.lon = lon

    def nearestVal(A, val):
        return A[np.argmin(numpy.abs(np.array(A)-val))]
    
    def run(self, dataslice):

        #need to calculate the local midnight for each night, and also twilight times
        unights = np.unique(dataSlice[self.nightcol])
        twilights = np.zeros([len(unights), 6]) #setting and rising.  
        localMidnight = np.zeros(len(unights))
        dataSlice.sort(order=self.mdjcol)
        left = np.searchsorted(dataSlice[self.nightcol], unight)
        mjds = dataSlice[mddcol][left] #mjds to use when calculating twilights and midnights
        lsstObs = ephem.Observer()
        lsstObs.lat = self.lat
        lsstObs.lon = self.lon
        horizons = ['-6', '-12', '-18']
        obsList = []
        S = ephem.Sun()
        for h in horizons:
            obs = ephem.Observer()
            obs.lat, obs.lon = self.lat, self.lon
            obs.horizon = h
            obsList.append(obs)
        for i,n in enumerate(unights):
            localMidnight[i] = nearestVal([lsstObs.previous_antitransit(S, start=mjds[i], use_center=True),
                                           lsstObs.next_antitransit(S, start=mjds[i], use_center=True)], mjds[i] )
            for j, obs in ennumerate(obsList):
                twilights[i,j] = nearestVal([obs.previous_rising(S, start=mjds[i], use_center=True),
                                           obs.next_rising(S, start=mjds[i], use_center=True) ], mjds[i] )
                twilights[i,j+3] = nearestVal([obs.previous_setting(S, start=mjds[i], use_center=True),
                                           obs.next_setting(S, start=mjds[i], use_center=True)], mjds[i])
            
                                          
                                       
        
        #colorDict = {'u':,'g':,'r':,'i':,'z':,'y':}
        
        #lsstObs.previous_rising(ephem.Sun(), start= , use_center=True) #to get 6 degree twilight

        #plan:
        # take the 1st observation, compute next and previous antisetting of sun, take the closest as the midnight for that day.  
        #loop through each night, compute the 3 twilights and midnight times.
        #plot the twilights
        #loop though each unique night:
        #    np.where that night is
        #    yaxis_time = (expmjd[good] - local midnight on that night)*24.
        #    identify break points as where the filter changes, or there's a large gap in observing time
        #  plot between break points with color from colorDict.  Don't use all the points to try and keep size down!
