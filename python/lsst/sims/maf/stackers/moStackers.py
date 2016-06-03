import numpy as np
from .baseStacker import BaseStacker
import warnings


__all__ = ['AppMagStacker', 'MagLimitStacker', 'SNRStacker', 'VisStacker',
           'EclStacker', 'AllMoStackers']

class AppMagStacker(BaseStacker):
    """
    Add the apparent magnitude of an object with a given Hval to the observations dataframe.
    """
    def __init__(self, magFilterCol='magFilter'):
        self.magFilterCol = magFilterCol
        self.colsReq = [self.magFilterCol]
        self.colsAdded = ['appMag']
        self.units = ['mag']

    def run(self, ssoObs, Href, Hval):
        ssoObs = self._addStackers(ssoObs)
        return self._run(ssoObs, Href=Href, Hval=Hval)

    def _run(self, ssoObs, Href=None, Hval=None):
        ssoObs['appMag'] = ssoObs[self.magFilterCol] + Hval - Href
        return ssoObs


class MagLimitStacker(BaseStacker):
    """
    Add the apparent magnitude limit with trailing or detection losses to the observations dataframe.
    """
    def __init__(self, m5Col='fiveSigmaDepth', lossCol='dmagDetect'):
        self.m5Col = m5Col
        self.lossCol = lossCol
        self.colsReq = [self.m5Col, self.lossCol]
        self.colsAdded = ['magLimit']
        self.units = ['mag']

    def run(self, ssoObs, Href=None, Hval=None):
        ssoObs = self._addStackers(ssoObs)
        return self._run(ssoObs, Href=Href, Hval=Hval)

    def _run(self, ssoObs, Href=None, Hval=None):
        ssoObs['magLimit'] = ssoObs[self.m5Col] - ssoObs[self.lossCol]
        return ssoObs

class SNRStacker(BaseStacker):
    """
    Add the SNR to the observations dataframe.
    """
    def __init__(self, magLimitCol='magLimit', appMagCol='appMag',gamma=0.038):
        self.appMagCol = appMagCol
        self.magLimitCol = magLimitCol
        self.colsReq = [self.appMagCol, self.magLimitCol]
        self.colsAdded = ['SNR']
        self.gamma = gamma
        self.units = ['SNR']

    def run(self, ssoObs, Href=None, Hval=None):
        ssoObs = self._addStackers(ssoObs)
        return self._run(ssoObs, Href=Href, Hval=Hval)

    def _run(self, ssoObs, Href=None, Hval=None):
        xval = np.power(10, 0.5*(ssoObs[self.appMagCol] - ssoObs[self.magLimitCol]))
        ssoObs['SNR'] = 1.0 / np.sqrt((0.04 - self.gamma)*xval + self.gamma*xval*xval)
        return ssoObs

class VisStacker(BaseStacker):
    """
    Calculate whether an object is visible according to
    Fermi-Dirac completeness formula (see SDSS, eqn 24, Stripe82 analysis:
    http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf).
    Calculate estimated completeness/probability of detection,
    then evaluates if this object could be visible.
    """
    def __init__(self, magLimitCol='magLimit',
                appMagCol='appMag', sigma=0.12):
        self.magLimitCol = magLimitCol
        self.appMagCol = appMagCol
        self.sigma = sigma
        self.colsReq = [self.magLimitCol, self.appMagCol]
        self.colsAdded = ['vis']
        self.units = ['']

    def run(self, ssoObs, Href=None, Hval=None):
        ssoObs = self._addStackers(ssoObs)
        return self._run(ssoObs, Href=Href, Hval=Hval)

    def _run(self, ssoObs, Href=None, Hval=None):
        completeness = 1.0 / (1 + np.exp((ssoObs[self.appMagCol] - ssoObs[self.magLimitCol])/self.sigma))
        probability = np.random.random_sample(len(ssoObs[self.appMagCol]))
        ssoObs['vis'] = np.where(probability <= completeness, 1, 0)
        return ssoObs

class EclStacker(BaseStacker):
    """
    Add ecliptic latitude/longitude to ssoObs (in degrees).
    """
    def __init__(self, raCol='ra', decCol='dec', inDeg=True):
        self.raCol = raCol
        self.decCol = decCol
        self.inDeg = inDeg
        self.colsReq = [self.raCol, self.decCol]
        self.colsAdded = ['ecLat', 'ecLon']
        self.units = ['deg', 'deg']
        self.ecnode = 0.0
        self.ecinc = np.radians(23.439291)

    def run(self, ssoObs, Href=None, Hval=None):
        ssoObs = self._addStackers(ssoObs)
        return self._run(ssoObs, Href=Href, Hval=Hval)

    def _run(self, ssoObs, Href=None, Hval=None):
        ra = ssoObs[self.raCol]
        dec = ssoObs[self.decCol]
        if self.inDeg:
            ra = np.radians(ra)
            dec = np.radians(dec)
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
        xp = x
        yp = np.cos(self.ecinc)*y + np.sin(self.ecinc)*z
        zp = -np.sin(self.ecinc)*y + np.cos(self.ecinc)*z
        ssoObs['ecLat'] = np.degrees(np.arcsin(zp))
        ssoObs['ecLon'] = np.degrees(np.arctan2(yp, xp))
        ssoObs['ecLon'] = ssoObs['ecLon'] % 360
        return ssoObs

class AllMoStackers(BaseStacker):
    """
    Since for moving objects we usually want to run all of these at once/in order,
    provide a convenient way to do it.
    """
    def __init__(self, appMagStacker=None, magLimitStacker=None, snrStacker=None,
                 visStacker=None, eclStacker=None):
        if appMagStacker is not None:
            self.appMag = appMagStacker
        else:
            self.appMag = AppMagStacker()
        if magLimitStacker is not None:
            self.magLimit = magLimitStacker
        else:
            self.magLimit = MagLimitStacker()
        if snrStacker is not None:
            self.snr = snrStacker
        else:
            self.snr = SNRStacker()
        if visStacker is not None:
            self.vis = visStacker
        else:
            self.vis = VisStacker()
        if eclStacker is not None:
            self.ec = eclStacker
        else:
            self.ec = EclStacker()
        # Grab all the columns added/required.
        self.colsAdded = []
        self.colsReq = []
        self.units = []
        for s in (self.appMag, self.magLimit, self.snr, self.vis, self.ec):
            self.colsAdded += s.colsAdded
            self.colsReq += s.colsReq
            self.units += s.units
        # Remove duplicates.
        self.colsAdded = list(set(self.colsAdded))
        self.colsReq = list(set(self.colsReq))

    def run(self, ssoObs, Href, Hval=None):
        if Hval is None:
            Hval = Href
        if len(ssoObs) == 0:
            return ssoObs
        # Add the columns.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ssoObs = self._addStackers(ssoObs)
        # Run the individual stackers, without individually adding columns.
        ssoObs = self.ec._run(ssoObs, Href, Hval)
        ssoObs = self.appMag._run(ssoObs, Href, Hval)
        ssoObs = self.magLimit._run(ssoObs, Href, Hval)
        ssoObs = self.snr._run(ssoObs, Href, Hval)
        ssoObs = self.vis._run(ssoObs, Href, Hval)
        return ssoObs
