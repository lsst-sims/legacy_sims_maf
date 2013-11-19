# Calculates the coadded m5 of the values at a gridpoint.
import numpy as np
from baseMetric import BaseMetric

class Coaddm5Metric(BaseMetric):
    """Calculate the minimum of a simData column."""
    def __init__(self, m5col = '5sigma_modified', metricName = 'coaddm5'):
        """Instantiate metric.
        m5col should be the column name of the individual visit m5 data. """
        if hasattr(m5col, '__iter__'):
            raise Exception('m5col should be single column name: %s' %(m5col))
        self.colname = m5col   
        super(Coaddm5Metric, self).__init__(self.colname, metricName)
        return
            
    def run(self, dataSlice):
        """Calculate the coadded m5 value at this point."""
        coaddm5 = 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.colname])))
        return coaddm5
