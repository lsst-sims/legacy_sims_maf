import numpy as np
"""Some simple functions that are useful for astrometry calculations. """

def sigma_slope(x,sigma_y): #move this inside the class to use badval, or just punt a nan?
    """For fitting a line, the uncertainty in the slope
       is given by the spread in x values and the uncertainties
       in the y values.  Resulting units are x/sigma_y"""
    w = 1./sigma_y**2
    denom = np.sum(w)*np.sum(w*x**2)-np.sum(w*x)**2
    if denom <= 0:
        return np.nan
    else:
        result = np.sqrt(np.sum(w)/denom )
        return result

def m52snr(m,m5):
    """find the SNR for a star of magnitude m obsreved
    under conditions of 5-sigma limiting depth m5.  This assumes
    Gaussianity and might not be strictly true in bluer filters.
    See table 2 and eq 5 in astroph/0805.2366 """
    snr = 5.*10.**(-0.4*(m-m5))
    return snr

def astrom_precision(fwhm,snr):
    """approx precision of astrometric measure given seeing and SNR """
    result = fwhm/(snr) #sometimes a factor of 2 in denomenator, whatever.  
    return result
