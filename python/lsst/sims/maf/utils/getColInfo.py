import numpy as np
import addCols 

class ColInfo(object):
    def __init__(self):
        self.defaultDataSource = None
        self.defaultUnit = ''
        
    def getUnits(self, colName):
        """Given a column name from OpSim, return appropriate units (for plot labels)."""
        unitDict = {'fieldID': '#',
                    'filter': 'filter',
                    'seqnNum' : '#',
                    'expMJD': 'MJD',
                    'expTime': 's',
                    'slewTime': 's',
                    'slewDist': 'rad',
                    'rotSkyPos': 'rad',
                    'rotTelPos': 'rad',
                    'rawSeeing': 'arcsec',
                    'finSeeing': 'arcsec', 
                    'seeing': 'arcsec',
                    'airmass': 'airmass',
                    'night': 'night',
                    'fieldRA': 'rad',
                    'fieldDec': 'rad', 
                    'hexdithra': 'rad', 
                    'hexdithdec': 'rad',
                    'moonRA': 'rad',
                    'moonDec': 'rad',
                    'moonAlt': 'rad',
                    'dist2Moon': 'rad', 
                    'VskyBright': 'mag/sq arcsec',
                    'perry_skybrightness': 'mag/sq arcsec',
                    'skybrightness_modified': 'mag/sq arcsec',
                    '5sigma': 'mag',
                    '5sigma_modified':'mag',
                    '5sigma_ps' : 'mag'}
        unitDict['normairmass'] = 'airmass/(minimum possible airmass)'
        unitDict['ra_pi_amp'] = 'arcsec'
        unitDict['dec_pi_amp'] = 'arcsec'
        if colName in unitDict:
            return unitDict[colName]
        else:
            return self.defaultUnit

    def getDataSource(self, colName):
        """Given a column name to be added to simdata, identify appropriate source. 

        For values from database, this is self.defaultDataSource ('db'). 
        For values which are precalculated for a particular column, this should be a 
        method added to this class."""
        # Note that a 'unique' list of methods should be built from the resulting returned
        #  methods, at whatever point the derived data columns will be calculated.
        sourceDict = {'normairmass': addCols.NormAirmass(),
                      'ra_pi_amp': addCols.ParallaxFactor(),
                      'dec_pi_amp': addCols.ParallaxFactor()}
        if colName in sourceDict:
            return sourceDict[colName]
        else:
            return self.defaultDataSource

