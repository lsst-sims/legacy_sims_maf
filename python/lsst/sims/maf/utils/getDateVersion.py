import os
import time

def getDateVersion():
    """return a string with today's date and a dict with the MAF version info """
    version_file = os.environ['SIMS_MAF_DIR']+'/python/lsst/sims/maf/'+'version.py'
    execfile(version_file, globals()) # Creates variables: __all__ = ('__version__', '__repo_version__', '__repo_version__', '__fingerprint__', '__dependency_versions__')
    today_date = time.strftime("%x")
    versionInfo = {'__version__':__version__,'__repo_version__':__repo_version__, '__fingerprint__':__fingerprint__, '__dependency_versions__':__dependency_versions__}

    return today_date, versionInfo

