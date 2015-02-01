import numpy as np
from .baseMetric import BaseMetric

class NChangesMetric(BaseMetric):
    """
    Compute the number of times a column value changes.
    (useful for filter changes in particular).
    """
    def __init__(self, col='filter', orderBy='expMJD', **kwargs):
        self.col = col
        self.orderBy = orderBy
        super(NChangesMetric, self).__init__(col=[col, orderBy], **kwargs)

    def run(self, dataSlice, slicePoint=None):
        idxs = np.argsort(dataSlice[self.orderBy])
        diff = (dataSlice[self.col][idxs][1:] != dataSlice[self.col][idxs][:-1])
        return len(np.where(diff == True)[0])

class OpenShutterFractionMetric(BaseMetric):
    """
    Compute the fraction of time the shutter is open compared to the total time spent observing.
    """
    def __init__(self, metricName='OpenShutterFraction',
                 slewTimeCol='slewTime', expTimeCol='visitExpTime', visitTimeCol='visitTime',
                 **kwargs):
        self.expTimeCol = expTimeCol
        self.visitTimeCol = visitTimeCol
        self.slewTimeCol = slewTimeCol
        super(OpenShutterFractionMetric, self).__init__(col=[self.expTimeCol, self.visitTimeCol, self.slewTimeCol],
                                                        metricName=metricName, units='OpenShutter/TotalTime',
                                                        **kwargs)
        if self.displayDict['group'] == 'Ungrouped':
            self.displayDict['group'] = 'Technical'
        if self.displayDict['caption'] is None:
            self.displayDict['caption'] = 'Open shutter time (%s total) divided by (total visit time (%s) + slewtime (%s)).' \
              %(self.expTimeCol, self.visitTimeCol, self.slewTimeCol)

    def run(self, dataSlice, slicePoint=None):
        result = (np.sum(dataSlice[self.expTimeCol])
                    / np.sum(dataSlice[self.slewTimeCol] + dataSlice[self.visitTimeCol]))
        return result

class CompletenessMetric(BaseMetric):
    """Compute the completeness and joint completeness """
    def __init__(self, filterColName='filter', metricName='Completeness',
                 u=0, g=0, r=0, i=0, z=0, y=0, **kwargs):
        """
        Compute the completeness for the each of the given filters and the
        joint completeness across all filters.

        Completeness calculated in any filter with a requested 'nvisits' value greater than 0, range is 0-1.
        """
        self.filterCol = filterColName
        super(CompletenessMetric,self).__init__(col=self.filterCol, metricName=metricName, **kwargs)
        self.nvisitsRequested = np.array([u, g, r, i, z, y])
        self.filters = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        # Remove filters from consideration where number of visits requested is zero.
        good = np.where(self.nvisitsRequested > 0)
        self.nvisitsRequested = self.nvisitsRequested[good]
        self.filters = self.filters[good]
        # Raise exception if number of visits wasn't changed from the default, for at least one filter.
        if len(self.filters) == 0:
            raise ValueError('Please set the requested number of visits for at least one filter.')
        # Set an order for the reduce functions (for display purposes only).
        for i, f in enumerate(('u', 'g', 'r', 'i', 'z', 'y', 'Joint')):
            self.reduceOrder[f] = i
        if self.displayDict['group'] == 'Ungrouped':
            self.displayDict['group'] = 'Technical'
        if self.displayDict['caption'] is None:
            self.displayDict['caption'] = 'Completeness fraction for each filter (and joint across all filters).'
            self.displayDict['caption'] += ' Calculated as number of visits compared to a benchmark value of:'
            for i, f in enumerate(self.filters):
                self.displayDict['caption'] += ' %s: %d' %(f, self.nvisitsRequested[i])
            self.displayDict['caption'] += '.'

    def run(self, dataSlice, slicePoint=None):
        """
        Compute the completeness for each filter, and then the minimum (joint) completeness for each slice.
        """
        allCompleteness = []
        for f, nVis in zip(self.filters, self.nvisitsRequested):
            filterVisits = np.size(np.where(dataSlice[self.filterCol] == f)[0])
            allCompleteness.append(filterVisits/np.float(nVis))
        allCompleteness.append(np.min(np.array(allCompleteness)))
        return np.array(allCompleteness)

    def reduceu(self, completeness):
        if 'u' in self.filters:
            return completeness[np.where(self.filters == 'u')[0]]
        else:
            return 1
    def reduceg(self, completeness):
        if 'g' in self.filters:
            return completeness[np.where(self.filters == 'g')[0]]
        else:
            return 1
    def reducer(self, completeness):
        if 'r' in self.filters:
            return completeness[np.where(self.filters == 'r')[0]]
        else:
            return 1
    def reducei(self, completeness):
        if 'i' in self.filters:
            return completeness[np.where(self.filters == 'i')[0]]
        else:
            return 1
    def reducez(self, completeness):
        if 'z' in self.filters:
            return completeness[np.where(self.filters == 'z')[0]]
        else:
            return 1
    def reducey(self, completeness):
        if 'y' in self.filters:
            return completeness[np.where(self.filters == 'y')[0]]
        else:
            return 1
    def reduceJoint(self, completeness):
        """
        The joint completeness is just the minimum completeness for a point/field.
        """
        return completeness[-1]


class VisitFiltersMetric(BaseMetric):
    """
    Calculate an RGBA value that accounts for the filters used up to time t0.
    """
    def __init__(self, filterCol='filter', timeCol='expMJD', t0=None, tStep=30./60./60./24., **kwargs):
        self.filter_rgba_map = {'u':(0,0,1),   #dark blue
                                'g':(0,1,1),  #cyan
                                'r':(0,1,0),    #green
                                'i':(1,0.5,0.3),  #orange
                                'z':(1,0,0),    #red
                                'y':(1,0,1)}  #magenta
        self.filterCol = filterCol
        self.timeCol = timeCol
        self.t0 = t0
        if self.t0 is None:
            self.t0 = 52939
        self.tStep = tStep
        super(VisitFiltersMetric, self).__init__(col=[filterCol, timeCol], **kwargs)
        self.metricDtype = 'object'
        self.plotDict['logScale'] = False
        self.plotDict['colorMax'] = 10
        self.plotDict['colorMin'] = 0
        self.plotDict['cbar'] = False
        self.plotDict['metricIsColor'] = True

    def _calcColor(self, filters):
        colorR = []
        colorG = []
        colorB = []
        for f in filters:
            color = self.filter_rgba_map[f]
            colorR.append(color[0])
            colorG.append(color[1])
            colorB.append(color[2])
        colorR = np.array(colorR, float)
        colorG = np.array(colorG, float)
        colorB = np.array(colorB, float)
        return colorR, colorG, colorB

    def _scaleColor(self, colorR, colorG, colorB):
        r = colorR.sum()
        g = colorG.sum()
        b = colorB.sum()
        scale = 1. / np.max([r, g, b])
        r *= scale
        g *= scale
        b *= scale
        return r, g, b

    def run(self, dataSlice, slicePoint=None):
        dts = np.abs(self.t0 - dataSlice[self.timeCol])
        visitNow = np.where(dts < self.tStep)[0]
        if len(visitNow) > 0:
            # We have some exact matches to this timestep, so just use their colors directly.
            colorR, colorG, colorB = self._calcColor(dataSlice[self.filterCol][visitNow])
            r, g, b = self._scaleColor(colorR, colorG, colorB)
            alpha = 1.0
        else:
            colorR, colorG, colorB = self._calcColor(dataSlice[self.filterCol])
            timeweight = dts.min()/dts
            r, g, b = self._scaleColor(colorR*timeweight, colorG*timeweight, colorB*timeweight)
            # These values for calculating alpha (the transparency of the final plotted point)
            #  are just numbers that seemed to make nice movies in my trials.
            # The exponential decay with the most recent time of observations (dts.min) gives a nice fast fade,
            #  and adding the len(dts) means that repeated observations show up a bit darker.
            # 0.8, 100, 50 and 0.14 are just empirically determined .. 0.14 will be the minimum transparency,
            #   and 0.9 will be the maximum. These were chosen to separate the peak from the 'active' observations,
            #   and not let the long-ago observations fade out too much. 
            alpha = np.max([0.8*np.exp(-100.*dts.min()+len(dts)/50.), 0.14])
            alpha = np.min([alpha, 0.9])
        return (r, g, b, alpha)
