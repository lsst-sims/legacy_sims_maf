
def getUnits(colName):
    """Keep track of units in Opsim columns for labeling plots """
    unitDict = dict(fieldID='#', filter='filter',  seqnNum='#',
                    expMJD='MJD',  expTime='s',slewTime='s',  slewDist='rad',
                    rotSkyPos='rad',rotTelPos='rad',  maxSeeing='arcsec',
                    rawSeeing='arcsec', seeing='arcsec',  xparency='',
                    cldSeeing='arcsec', airmass='airmass', VskyBright='mag/sq arcsec',
                    fieldRA='rad', fieldDec='rad', dist2Moon='rad',
                    moonRA='rad', moonDec='rad', moonAlt='rad',
                    perry_skybrightness='mag/sq arcsec',
                    skybrightness_modified='mag/sq arcsec' ,
                    night='night',
                    hexdithra='rad',       hexdithdec='rad'  
        )
    unitDict['5sigma'] = 'mag'
    unitDict['5sigma_ps'] = 'mag'
    unitDict['5sigma_modified'] = 'mag'
    unitDict['normairmass'] = 'airmass/(minimum possible airmass)'
    try:
        result = unitDict[colName]
    except:
        result=None
    return result
