import os
import time
import lsst.sims.maf

__all__ = ['getDateVersion']

def getDateVersion():
    """
    Return a string with today's date and a dict with the MAF version info.
    """

    version = lsst.sims.maf.version
    today_date = time.strftime("%x")
    versionInfo = {'__version__':version.__version__,
                   '__repo_version__':version.__repo_version__,
                   '__fingerprint__':version.__fingerprint__,
                   '__dependency_versions__':version.__dependency_versions__}

    return today_date, versionInfo
