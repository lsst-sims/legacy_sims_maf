import numpy as np
import healpy as hp
from scipy import interpolate
from .baseMetric import BaseMetric

# A collection of metrics which are primarily intended to be used as summary statistics.

__all__ = ['fOArea', 'fONv', 'TableFractionMetric', 'IdentityMetric',
           'NormalizeMetric', 'ZeropointMetric', 'TotalPowerMetric',
           'StaticProbesFoMEmulatorMetricSimple']


class fONv(BaseMetric):
    """
    Metrics based on a specified area, but returning NVISITS related to area:
    given Asky, what is the minimum and median number of visits obtained over that much area?
    (choose the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Asky : float, opt
        Area of the sky to base the evaluation of number of visits over.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    Nvisit : int, opt
        Number of visits to use as the benchmark value, if choosing to return a normalized Nvisit value.
    norm : boolean, opt
        Normalize the returned "nvisit" (min / median) values by Nvisit, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fONv.
    """
    def __init__(self, col='metricdata', Asky=18000., nside=128, Nvisit=825,
                 norm=False, metricName='fONv',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        # Determine how many healpixels are included in Asky sq deg.
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.npix_Asky = np.int(np.ceil(self.Asky / self.scale))
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        result = np.empty(2, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'][0] = "MedianNvis"
        result['name'][1] = "MinNvis"
        # If there is not even as much data as needed to cover Asky:
        if len(dataSlice) < self.npix_Asky:
            # Return the same type of metric value, to make it easier downstream.
            result['value'][0] = self.badval
            result['value'][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis_Asky = nvis_sorted[-self.npix_Asky:]
        result['value'][0] = np.median(nvis_Asky)
        result['value'][1] = np.min(nvis_Asky)
        if self.norm:
            result['value'] /= float(self.Nvisit)
        return result


class fOArea(BaseMetric):
    """
    Metrics based on a specified number of visits, but returning AREA related to Nvisits:
    given Nvisit, what amount of sky is covered with at least that many visits?

    Parameters
    ----------
    col : str or list of strs, opt
        Name of the column in the numpy recarray passed to the summary metric.
    Nvisit : int, opt
        Number of visits to use as the minimum required -- metric calculated area that has this many visits.
        Default 825.
    Asky : float, opt
        Area to use as the benchmark value, if choosing to returned a normalized Area value.
        Default 18,0000 sq deg.
    nside : int, opt
        Nside parameter from healpix slicer, used to set the physical relationship between on-sky area
        and number of healpixels. Default 128.
    norm : boolean, opt
        Normalize the returned "area" (area with minimum Nvisit visits) value by Asky, if true.
        Default False.
    metricName : str, opt
        Name of the summary metric. Default fOArea.
    """
    def __init__(self, col='metricdata', Nvisit=825, Asky = 18000.0, nside=128,
                  norm=False, metricName='fOArea',  **kwargs):
        """Asky = square degrees """
        super().__init__(col=col, metricName=metricName, **kwargs)
        self.Nvisit = Nvisit
        self.nside = nside
        self.Asky = Asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, dataSlice, slicePoint=None):
        name = dataSlice.dtype.names[0]
        nvis_sorted = np.sort(dataSlice[name])
        # Identify the healpixels with more than Nvisits.
        nvis_min = nvis_sorted[np.where(nvis_sorted >= self.Nvisit)]
        if len(nvis_min) == 0:
            result = self.badval
        else:
            result = nvis_min.size * self.scale
            if self.norm:
                result /= float(self.Asky)
        return result


class TableFractionMetric(BaseMetric):
    """
    Count the completeness (for many fields) and summarize how many fields have given completeness levels
    (within a series of bins). Works with completenessMetric only.

    This metric is meant to be used as a summary statistic on something like the completeness metric.
    The output is DIFFERENT FROM SSTAR and is:
    element   matching values
    0         0 == P
    1         0 < P < .1
    2         .1 <= P < .2
    3         .2 <= P < .3
    ...
    10        .9 <= P < 1
    11        1 == P
    12        1 < P
    Note the 1st and last elements do NOT obey the numpy histogram conventions.
    """
    def __init__(self, col='metricdata',  nbins=10, maskVal=0.):
        """
        colname = the column name in the metric data (i.e. 'metricdata' usually).
        nbins = number of bins between 0 and 1. Should divide evenly into 100.
        """
        super(TableFractionMetric, self).__init__(col=col, maskVal=maskVal, metricDtype='float')
        self.nbins = nbins

    def run(self, dataSlice, slicePoint=None):
        # Calculate histogram of completeness values that fall between 0-1.
        goodVals = np.where((dataSlice[self.colname] > 0) & (dataSlice[self.colname] < 1)  )
        bins = np.arange(self.nbins+1.)/self.nbins
        hist, b = np.histogram(dataSlice[self.colname][goodVals], bins=bins)
        # Fill in values for exact 0, exact 1 and >1.
        zero = np.size(np.where(dataSlice[self.colname] == 0)[0])
        one = np.size(np.where(dataSlice[self.colname] == 1)[0])
        overone = np.size(np.where(dataSlice[self.colname] > 1)[0])
        hist = np.concatenate((np.array([zero]), hist, np.array([one]), np.array([overone])))
        # Create labels for each value
        binNames = ['0 == P']
        binNames.append('0 < P < 0.1')
        for i in np.arange(1, self.nbins):
            binNames.append('%.2g <= P < %.2g'%(b[i], b[i+1]) )
        binNames.append('1 == P')
        binNames.append('1 < P')
        # Package the names and values up
        result = np.empty(hist.size, dtype=[('name', np.str_, 20), ('value', float)])
        result['name'] = binNames
        result['value'] = hist
        return result


class IdentityMetric(BaseMetric):
    """
    Return the metric value itself .. this is primarily useful as a summary statistic for UniSlicer metrics.
    """
    def run(self, dataSlice, slicePoint=None):
        if len(dataSlice[self.colname]) == 1:
            result = dataSlice[self.colname][0]
        else:
            result = dataSlice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'normVal'. Useful for turning summary statistics into fractions.
    """
    def __init__(self, col='metricdata', normVal=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.normVal = float(normVal)
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname]/self.normVal
        if len(result) == 1:
            return result[0]
        else:
            return result

class ZeropointMetric(BaseMetric):
    """
    Return a metric values with the addition of 'zp'. Useful for altering the zeropoint for summary statistics.
    """
    def __init__(self, col='metricdata', zp=0, **kwargs):
        super(ZeropointMetric, self).__init__(col=col, **kwargs)
        self.zp = zp
    def run(self, dataSlice, slicePoint=None):
        result = dataSlice[self.colname] + self.zp
        if len(result) == 1:
            return result[0]
        else:
            return result

class TotalPowerMetric(BaseMetric):
    """
    Calculate the total power in the angular power spectrum between lmin/lmax.
    """
    def __init__(self, col='metricdata', lmin=100., lmax=300., removeDipole=True, maskVal=hp.UNSEEN, **kwargs):
        self.lmin = lmin
        self.lmax = lmax
        self.removeDipole = removeDipole
        super(TotalPowerMetric, self).__init__(col=col, maskVal=maskVal, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Calculate the power spectrum.
        if self.removeDipole:
            cl = hp.anafast(hp.remove_dipole(dataSlice[self.colname], verbose=False))
        else:
            cl = hp.anafast(dataSlice[self.colname])
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition]*(2*ell[condition]+1))
        return totalpower


class StaticProbesFoMEmulatorMetric(BaseMetric):
    """This calculates the Figure of Merit for the combined
    static probes (3x2pt, i.e., Weak Lensing, LSS, Clustering).
    This FoM takes into account the effects of the following systematics:
        - multiplicative shear bias
        - intrinsic alignments
        - galaxy bias
        - baryonic physics effects
        - photometric redshift uncertainties
    Default values for these systematics are provided
    
    The Emulator is uses a Gaussian Process to effectively interpolate between 
    a grid of FoM values. 
    """
    def __init__(self, nside=128, year=10, 
                 shear_m=0.003, sigma_z=0.05, sig_delta_z=0.001, sig_sigma_z=0.003,
                 col=None, **kwargs):
        
        """
        Args:
            nside (int): healpix resolution
            year (int): year of the FoM emulated values, 
                can be one of [1, 3, 6, 10]
            col (str): column name of metric data.
        """
        self.nside = nside
        super().__init__(col=col, **kwargs)
        if col is None:
            self.col = 'metricdata'
        self.year = year
        self.shear_m = shear_m
        self.sigma_z = sigma_z
        self.sig_delta_z = sig_delta_z
        self.sig_sigma_z = sig_sigma_z

    def run(self, dataSlice, slicePoint=None):
        """
        Args:
            dataSlice (ndarray): Values passed to metric by the slicer, 
                which the metric will use to calculate metric values 
                at each slicePoint.
            slicePoint (Dict): Dictionary of slicePoint metadata passed
                to each metric.
        Returns:
             float: Interpolated static-probe statistical Figure-of-Merit.
        Raises:
             ValueError: If year is not one of the 4 for which a FoM is calculated
        """
        # Chop off any outliers
        good_pix = np.where(dataSlice[self.col] > 0)[0]
        
        # Calculate area and med depth from
        area = hp.nside2pixarea(self.nside, degrees=True) * np.size(good_pix)
        median_depth = np.median(dataSlice[self.col][good_pix])

        # FoM is calculated at the following values
        parameters = dict(
            area = np.array([ 7623.22, 14786.3 ,  9931.47,  8585.43, 17681.8 , 15126.9 ,
               9747.99,  8335.08,  9533.42, 18331.3 , 12867.8 , 17418.9 ,
               19783.1 , 12538.8 , 15260.  , 16540.7 , 19636.8 , 11112.7 ,
               10385.5 , 16140.2 , 18920.1 , 17976.2 , 11352.  ,  9214.77,
               16910.7 , 11995.6 , 16199.8 , 14395.1 ,  8133.86, 13510.5 ,
               19122.3 , 15684.5 , 12014.8 , 14059.7 , 10919.3 , 13212.7 ]),
            depth = np.array([25.3975, 26.5907, 25.6702, 26.3726, 26.6691, 24.9882, 25.0814,
               26.4247, 26.5088, 25.5596, 25.3288, 24.8035, 24.8792, 25.609 ,
               26.2385, 25.0351, 26.7692, 26.5693, 25.8799, 26.3009, 25.5086,
               25.4219, 25.8305, 26.2953, 26.0183, 25.26  , 25.7903, 25.1846,
               26.7264, 26.0507, 25.6996, 25.2256, 24.9383, 26.1144, 25.9464,
               26.1878]),
            shear_m = np.array([0.00891915, 0.0104498 , 0.0145972 , 0.0191916 , 0.00450246,
               0.00567828, 0.00294841, 0.00530922, 0.0118632 , 0.0151849 ,
               0.00410151, 0.0170622 , 0.0197331 , 0.0106615 , 0.0124445 ,
               0.00994507, 0.0136251 , 0.0143491 , 0.0164314 , 0.016962  ,
               0.0186608 , 0.00945903, 0.0113246 , 0.0155225 , 0.00800846,
               0.00732104, 0.00649453, 0.00243976, 0.0125932 , 0.0182587 ,
               0.00335859, 0.00682287, 0.0177269 , 0.0035219 , 0.00773304,
               0.0134886 ]),
            sigma_z = np.array([0.0849973, 0.0986032, 0.0875521, 0.0968222, 0.0225239, 0.0718278,
               0.0733675, 0.0385274, 0.0425549, 0.0605867, 0.0178555, 0.0853407,
               0.0124119, 0.0531027, 0.0304032, 0.0503145, 0.0132213, 0.0941765,
               0.0416444, 0.0668198, 0.063227 , 0.0291332, 0.0481633, 0.0595606,
               0.0818742, 0.0472518, 0.0270185, 0.0767401, 0.0219945, 0.0902663,
               0.0779705, 0.0337666, 0.0362358, 0.0692429, 0.0558841, 0.0150457]),
            sig_delta_z = np.array([0.0032537 , 0.00135316, 0.00168787, 0.00215043, 0.00406031,
               0.00222358, 0.00334993, 0.00255186, 0.00266499, 0.00159226,
               0.00183664, 0.00384965, 0.00427765, 0.00314377, 0.00456113,
               0.00347868, 0.00487938, 0.00418152, 0.00469911, 0.00367598,
               0.0028009 , 0.00234161, 0.00194964, 0.00200982, 0.00122739,
               0.00310886, 0.00275168, 0.00492736, 0.00437241, 0.00113931,
               0.00104864, 0.00292328, 0.00452082, 0.00394114, 0.00150756,
               0.003613  ]),
            sig_sigma_z= np.array([0.00331909, 0.00529541, 0.00478151, 0.00437497, 0.00443062,
               0.00486333, 0.00467423, 0.0036723 , 0.00426963, 0.00515357,
               0.0054553 , 0.00310132, 0.00305971, 0.00406327, 0.00594293,
               0.00348709, 0.00562526, 0.00396025, 0.00540537, 0.00500447,
               0.00318595, 0.00460592, 0.00412137, 0.00336418, 0.00524988,
               0.00390092, 0.00498349, 0.0056667 , 0.0036384 , 0.00455861,
               0.00554822, 0.00381061, 0.0057615 , 0.00357705, 0.00590572,
               0.00422393]),
            FOM = np.array([11.708, 33.778, 19.914, 19.499, 41.173, 17.942, 12.836, 26.318,
               25.766, 28.453, 28.832, 14.614, 23.8  , 21.51 , 27.262, 20.539,
               39.698, 19.342, 17.103, 25.889, 25.444, 32.048, 24.611, 23.747,
               32.193, 18.862, 34.583, 14.54 , 23.31 , 25.812, 39.212, 25.078,
               14.339, 24.12 , 24.648, 29.649])
        )
        
        # Standardizing data
        df_unscaled = pd.DataFrame(parameters)
        X_params = ['area', 'depth', 'shear_m', 'sigma_z', 'sig_delta_z', 'sig_sigma_z']
        scaler = StandardScaler()
        scalerX = StandardScaler()
        scalerY = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df_unscaled), columns=parameters.keys())
        X = df_unscaled.drop('FOM', axis=1)
        X = pd.DataFrame(scalerX.fit_transform(df_unscaled.drop('FOM', axis=1)), columns=X_params)
        Y = pd.DataFrame(scalerY.fit_transform(np.array(df_unscaled['FOM']).reshape(-1, 1)), columns=['FOM'])

        # Building Gaussian Process based emulator
        kernel = kernels.ExpSquaredKernel(metric=[1,1,1,1,1,1], ndim=6)
        gp = george.GP(kernel, mean=df['FOM'].mean())
        gp.compute(X) #I have taken the last raw put to use it as test point 

        def neg_ln_lik(p):
                    gp.set_parameter_vector(p)
                    return -gp.log_likelihood(df['FOM']) 
        def grad_neg_ln_like(p):
                    gp.set_parameter_vector(p)
                    return -gp.grad_log_likelihood(df['FOM']) 
        result = minimize(neg_ln_lik, gp.get_parameter_vector(), jac=grad_neg_ln_like)

        gp.set_parameter_vector(result.x)
        
        # Survey parameters to predict FoM at
        to_pred = np.array([[area, median_depth, self.shear_m, self.sigma_z, self.sig_delta_z, self.sig_sigma_z]])
        to_pred = scalerX.transform(to_pred)
        
        predSFOM = gp.predict(df['FOM'], to_pred, return_cov=False)
        
        predFOM = scalerY.inverse_transform(list(predSFOM))
        
        return predFOM[0]

