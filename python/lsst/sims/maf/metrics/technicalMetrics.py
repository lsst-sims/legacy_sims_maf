import numpy as np
from .baseMetric import BaseMetric

__all__ = ['NChangesMetric', 'MinDeltaTimeChangesMetric', 'DeltaTimeChangesMetric',
           'TeffMetric', 'OpenShutterFractionMetric',
           'CompletenessMetric', 'FilterColorsMetric']

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

class MinDeltaTimeChangesMetric(BaseMetric):
    """
    Compute the minimum time between changes in a column value.
    (useful for calculating time between filter changes in particular).
    Returns delta time in minutes!
    """
    def __init__(self, col='filter', timeCol='expMJD', **kwargs):
        self.col = col
        self.timeCol = timeCol
        super(MinDeltaTimeChangesMetric, self).__init__(col=[col, timeCol], **kwargs)

    def run(self, dataSlice, slicePoint=None):
        idxs = np.argsort(dataSlice[self.timeCol])
        diff = (dataSlice[self.col][idxs][1:] != dataSlice[self.col][idxs][:-1])
        condition = np.where(diff==True)[0]
        dtimes = dataSlice[self.timeCol][1:][condition] - dataSlice[self.timeCol][:-1][condition]
        return dtimes.min()*24.0*60.0

class DeltaTimeChangesMetric(BaseMetric):
    """
    Compute (all of) the time between changes in a column value.
    (useful for calculating time between filter changes in particular).
    Returns delta time in minutes!
    """
    def __init__(self, col='filter', timeCol='expMJD', **kwargs):
        self.col = col
        self.timeCol = timeCol
        super(DeltaTimeChangesMetric, self).__init__(col=[col, timeCol], **kwargs)

    def run(self, dataSlice, slicePoint=None):
        idxs = np.argsort(dataSlice[self.timeCol])
        diff = (dataSlice[self.col][idxs][1:] != dataSlice[self.col][idxs][:-1])
        condition = np.where(diff==True)[0]
        dtimes = dataSlice[self.timeCol][1:][condition] - dataSlice[self.timeCol][:-1][condition]
        return dtimes*24.0*60.0

class TeffMetric(BaseMetric):
    """
    Effective time equivalent for a given set of visits.
    """
    def __init__(self, m5Col='fiveSigmaDepth', filterCol='filter', metricName='tEff',
                 fiducialDepth=None, teffBase=30.0, **kwargs):
        self.m5Col = m5Col
        self.filterCol = filterCol
        if fiducialDepth is None:
            self.depth = {'u':23.9,'g':25.0, 'r':24.7, 'i':24.0, 'z':23.3, 'y':22.1} # design value
        else:
            if isinstance(fiducialDepth, dict):
                self.depth = fiducialDepth
            else:
                raise ValueError('fiducialDepth should be None or dictionary')
        self.teffBase = teffBase
        super(TeffMetric, self).__init__(col=[m5Col, filterCol], metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        filters = np.unique(dataSlice[self.filterCol])
        teff = 0.0
        for f in filters:
            match = np.where(dataSlice[self.filterCol] == f)[0]
            teff += (10.0**(0.8*(dataSlice[self.m5Col][match] - self.depth[f]))).sum()
        teff *= self.teffBase
        return teff

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


class FilterColorsMetric(BaseMetric):
    """
    Calculate an RGBA value that accounts for the filters used up to time t0.
    """
    def __init__(self, rRGB='rRGB', gRGB='gRGB', bRGB='bRGB',
                 timeCol='expMJD', t0=None, tStep=40./60./60./24.,
                 metricName='FilterColors', **kwargs):
        """
        t0 = the current time
        """
        self.rRGB = rRGB
        self.bRGB = bRGB
        self.gRGB = gRGB
        self.timeCol = timeCol
        self.t0 = t0
        if self.t0 is None:
            self.t0 = 52939
        self.tStep = tStep
        super(FilterColors, self).__init__(col=[rRGB, gRGB, bRGB, timeCol],
                                           metricName=metricName, **kwargs)
        self.metricDtype = 'object'
        self.plotDict['logScale'] = False
        self.plotDict['colorMax'] = 10
        self.plotDict['colorMin'] = 0
        self.plotDict['cbar'] = False
        self.plotDict['metricIsColor'] = True

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
        deltaT = np.abs(dataSlice[self.timeCol]-self.t0)
        visitNow = np.where(deltaT <= self.tStep)[0]
        if len(visitNow) > 0:
            # We have exact matches to this timestep, so use their colors directly and set alpha to >1.
            r, g, b = self._scaleColor(dataSlice[visitNow][self.rRGB],
                                       dataSlice[visitNow][self.gRGB],
                                       dataSlice[visitNow][self.bRGB])
            alpha = 10.
        else:
            # This part of the sky has only older exposures.
            deltaTmin = deltaT.min()
            nObs = len(dataSlice[self.timeCol])
            # Generate a combined color (weighted towards most recent observation).
            decay = deltaTmin/deltaT
            r, g, b = self._scaleColor(dataSlice[self.rRGB]*decay,
                                       dataSlice[self.gRGB]*decay,
                                       dataSlice[self.bRGB]*decay)
            # Then generate an alpha value, between alphamax/alphamid for visits
            #  happening within the previous 12 hours, then falling between
            #  alphamid/alphamin with a value that depends on the number of obs.
            alphamax = 0.8
            alphamid = 0.5
            alphamin = 0.2
            if deltaTmin < 0.5:
                alpha = np.exp(-deltaTmin*10.)*(alphamax - alphamid) + alphamid
            else:
                alpha = nObs/800.*alphamid
            alpha = np.max([alpha, alphamin])
            alpha = np.min([alphamax, alpha])
        return (r, g, b, alpha)
