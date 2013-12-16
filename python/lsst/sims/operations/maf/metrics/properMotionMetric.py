import numpy as np
from .baseMetric import BaseMetric



def sigma_slope(x,sigma_y):
    """For fitting a line, the uncertainty in the slope
       is given by the spread in x values and the uncertainties
       in the y values.  Resulting units are x/sigma_y"""
    w = 1./sigma_y**2
    denom = np.sum(w)*np.sum(w*x**2)-np.sum(w*x)**2
    result = np.sqrt(np.sum(w)/denom )
    return result

def m52snr(m,m5):
    """find the SNR for a star of magnitude m obsreved
    under conditions of 5-sigma limiting depth m5 """
    snr = 5.*10.**(-0.4*(m-m5))
    return snr

def astrom_precision(fwhm,snr):
    """approx precision of astrometric measure given seeing and SNR """
    result = fwhm/(snr) #sometimes a factor of 2 in denomenator, whatever.  
    return result

class ProperMotionMetric(BaseMetric):
    """Calculate the uncertainty in the returned proper
    motion.  Assuming Gaussian errors.
    """

    def __init__(self, metricName='properMotion',
                 m5col='5sigma_modified', mjdcol='expMJD',
                 filtercol='filter', seeingcol='seeing', u=16.,
                 g=16., r=16., i=16., z=16., y=16., badval= -666,
                 stellarType=None):
        """ Instantiate metric.

        m5col = column name for inidivual visit m5
        mjdcol = column name for exposure time dates
        filtercol = column name for filter
        seeingcol = column name for seeing (assumed FWHM)
        u,g,r,i,z = mag of fiducial star in each filter """
        cols = [m5col, mjdcol,filtercol,seeingcol]
        super(ProperMotionMetric, self).__init__(cols, metricName)
        # set return type
        self.metricDtype = 'float'
        #set units
        self.metricUnits = 'mas/yr'
        self.mags={'u':u, 'g':g,'r':r,'i':i,'z':z,'y':y}
        self.badval = badval
        if stellarType != None:
            raise NotImplementedError('Spot to add colors for different stars')

    def run(self, dataslice):
        filters = np.unique(dataslice['filter'])
        precis = np.zeros(dataslice.size, dtype='float')
        for f in filters:
            observations = np.where(dataslice['filter'] == f)
            if np.size(observations[0]) < 2:
                result = self.badval
            else:
                snr = m52snr(self.mags[f],
                   dataslice['5sigma_modified'][observations])
                precis[observations] = astrom_precision(
                    dataslice['seeing'][observations], snr)
                result = sigma_slope(dataslice['expMJD'][observations], precis)
                result = result*365.25*1e3 #convert to mas/yr
        # Observations that are very close together can still fail
        if np.isnan(result):
            result = self.badval 
        return result
        
