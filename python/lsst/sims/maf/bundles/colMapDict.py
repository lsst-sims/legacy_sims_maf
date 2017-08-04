__all__ = ['ColMapDict']


def ColMapDict(dictName=None):

    if dictName is None:
        dictName = 'opsimv4'
    dictName = dictName.lower()

    if dictName == 'opsimv4':
        colMap = {}
        colMap['ra'] = 'fieldRA'
        colMap['dec'] = 'fieldDec'
        colMap['mjd'] = 'observationStartMJD'
        colMap['exptime'] = 'visitExposureTime'
        colMap['visittime'] = 'visitTime'
        colMap['alt'] = 'altitude'
        colMap['az'] = 'azimuth'
        colMap['filter'] = 'filter'
        colMap['fiveSigmaDepth'] = 'fiveSigmaDepth'
        colMap['night'] = 'night'
        colMap['slewtime'] = 'slewTime'
        colMap['seeingEff'] = 'seeingFwhmEff'
        colMap['seeingGeom'] = 'seeingFwhmGeom'

    elif dictName == 'opsimv3':
        colMap = {}
        colMap['ra'] = 'fieldRA'
        colMap['dec'] = 'fieldDec'
        colMap['mjd'] = 'expMJD'
        colMap['exptime'] = 'visitExpTime'
        colMap['visittime'] = 'visitTime'
        colMap['alt'] = 'altitude'
        colMap['az'] = 'azimuth'
        colMap['filter'] = 'filter'
        colMap['fiveSigmaDepth'] = 'fiveSigmaDepth'
        colMap['night'] = 'night'
        colMap['slewtime'] = 'slewTime'
        colMap['seeingEff'] = 'FWHMeff'
        colMap['seeingGeom'] = 'FWHMgeom'

    else:
        raise ValueError('No built in column dict with that name.')

    return colMap