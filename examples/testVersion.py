import lsst.sims.maf.utils as utils

date, versionInfo = utils.getDateVersion()

print "Date", date
print "VersionInfo", versionInfo

print "Recorded version", versionInfo['__version__']
print "Recorded fingerprint", versionInfo['__fingerprint__']

