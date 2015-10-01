import numpy as np
import numpy.ma as ma
import warnings

from .moMetrics import BaseMoMetric

__all__ = ['ValueAtHMetric',
            'MO_CompletenessMetric', 'MO_CumulativeCompletenessMetric',
            'integrateOverH']

def integrateOverH(Mvalues, Hvalues, Hindex = 0.3):
    # Set expected H distribution.
    # dndh = differential size distribution (number in this bin)
    dndh = np.power(10., Hindex*(Hvalues-Hvalues.min()))
    # dn = cumulative size distribution (number in this bin and brighter)
    intVals = np.cumsum(Mvalues*dndh)/np.cumsum(dndh)
    return intVals

class ValueAtHMetric(BaseMoMetric):
    """
    Return the value of a metric at a given H.
    """
    def __init__(self, Hmark=22, **kwargs):
        metricName = 'Value At H=%.1f' %(Hmark)
        super(ValueAtHMetric, self).__init__(metricName=metricName, **kwargs)
        self.Hmark = Hmark

    def run(self, metricVals, Hvals):
        # Check if desired H value is within range of H values.
        if (self.Hmark < Hvals.min()) or (self.Hmark > Hvals.max()):
            warnings.warn('Desired H value of metric outside range of provided H values.')
            return None
        nHvals = len(Hvals)
        nHMetricVals = metricVals.shape[1]
        if nHvals == nHMetricVals:
            # Hvals matched the points where the metric values were calculated (clone H distribution).
            eps = 1.0e-6
            # Hvals is an array used for each metric value,
            #  we have to pick out the particular metricValues to use.
            diffs = np.abs(self.Hmark - Hvals)
            Hidx = np.where(diffs == diffs.min())[0]
            result = metricVals.swapaxes(0,1)[Hidx]
            Hmark = Hvals[Hidx]
            self.name = 'Value At H=%.1f' %(Hmark)
        else:
            # We have a range of metric values, one per Hval.
            result = np.interpolate([self.Hmark], Hvals, metricVals.swapaxes(0, 1))
        return result, Hmark

class MO_CompletenessMetric(BaseMoMetric):
    """
    Take the discoveryChances metric results and turn it into
    completeness estimate (relative to the entire population).
    Require at least 'requiredChances' to count an object as "found".
    """
    def __init__(self, requiredChances=1, nbins=20, minHrange=1.0,
                 metricName='Completeness', **kwargs):
        super(CompletenessMetric, self).__init__(metricName=metricName, **kwargs)
        self.units = '@ H'
        self.requiredChances = requiredChances
        # If H is not a cloned distribution, then we need to specify how to bin these values.
        self.nbins = nbins
        self.minHrange = minHrange

    def run(self, discoveryChances, Hvals):
        nSsos = discoveryChances.shape[0]
        nHval = len(Hvals)
        discoveriesH = discoveryChances.swapaxes(0, 1)
        if nHval == discoveryChances.shape[1]:
            # Hvals array is probably the same as the cloned H array.
            completeness = np.zeros(len(Hvals), float)
            for i, H in enumerate(Hvals):
                completeness[i] = np.where(discoveriesH[i].filled(0) >= self.requiredChances)[0].size
            completeness = completeness / float(nSsos)
        else:
            # The Hvals are spread more randomly among the objects (we probably used one per object).
            hrange = Hvals.max() - Hvals.min()
            minH = Hvals.min()
            if hrange < self.minHrange:
                hrange = self.minHrange
                minH = Hvals.min() - hrange/2.0
            stepsize = hrange / float(self.nbins)
            bins = np.arange(minH, minH + hrange + stepsize/2.0, stepsize)
            Hvals = bins[:-1]
            n_all, b = np.histogram(discoveriesH[0], bins)
            condition = np.where(discoveriesH[0] >= self.requiredChances)[0]
            n_found, b = np.histogram(discoveriesH[0][condition], bins)
            completeness = n_found.astype(float) / n_all.astype(float)
            completeness = np.where(n_all==0, 0, completeness)
        return completeness, Hvals


class MO_CumulativeCompletenessMetric(BaseMoMetric):
    """
    Take the discoveryChances metric results and turn it into
    completeness estimate (relative to the entire population).
    Require at least 'requiredChances' to count an object as "found".
    """
    def __init__(self, requiredChances=1, nbins=20, minHrange=1.0, Hindex=0.3, 
                 metricName='CumulativeCompleteness', **kwargs):
        super(CumulativeCompletenessMetric, self).__init__(metricName=metricName, **kwargs)
        self.units = '<= H'
        self.requiredChances = requiredChances
        # If H is not a cloned distribution, then we need to specify how to bin these values.
        self.nbins = nbins
        self.minHrange = minHrange
        self.Hindex = Hindex

    def run(self, discoveryChances, Hvals):
        nSsos = discoveryChances.shape[0]
        nHval = len(Hvals)
        discoveriesH = discoveryChances.swapaxes(0, 1)
        if nHval == discoveryChances.shape[1]:
            # Hvals array is probably the same as the cloned H array.
            completeness = np.zeros(len(Hvals), float)
            for i, H in enumerate(Hvals):
                completeness[i] = np.where(discoveriesH[i].filled(0) >= self.requiredChances)[0].size
            completeness = completeness / float(nSsos)
        else:
            # The Hvals are spread more randomly among the objects (we probably used one per object).
            hrange = Hvals.max() - Hvals.min()
            minH = Hvals.min()
            if hrange < self.minHrange:
                hrange = self.minHrange
                minH = Hvals.min() - hrange/2.0
            stepsize = hrange / float(self.nbins)
            bins = np.arange(minH, minH + hrange + stepsize/2.0, stepsize)
            Hvals = bins[:-1]
            n_all, b = np.histogram(discoveriesH[0], bins)
            condition = np.where(discoveriesH[0] >= self.requiredChances)[0]
            n_found, b = np.histogram(discoveriesH[0][condition], bins)
            completeness = n_found.astype(float) / n_all.astype(float)
            completeness = np.where(n_all==0, 0, completeness)
        completenessInt = integrateOverH(completeness, Hvals, self.Hindex)
        return completenessInt, Hvals
