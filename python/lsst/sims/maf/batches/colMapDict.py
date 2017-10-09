__all__ = ['ColMapDict']


def getColMap(opsdb):
    """Get the colmap dictionary, if you already have a database object.

    Parameters
    ----------
    opsdb : lsst.sims.maf.db.Database or lsst.sims.maf.db.OpsimDatabase

    Returns
    -------
    dictionary
    """
    try:
        version = opsdb.opsimversion
        version = 'opsim' + version.lower()
    except AttributeError:
        version = 'barebones'
    colmap = ColMapDict(version)
    return colmap


def ColMapDict(dictName=None):

    if dictName is None:
        dictName = 'opsimv4'
    dictName = dictName.lower()

    if dictName == 'opsimv4':
        colMap = {}
        colMap['ra'] = 'fieldRA'
        colMap['dec'] = 'fieldDec'
        colMap['raDecDeg'] = True
        colMap['mjd'] = 'observationStartMJD'
        colMap['exptime'] = 'visitExposureTime'
        colMap['visittime'] = 'visitTime'
        colMap['alt'] = 'altitude'
        colMap['az'] = 'azimuth'
        colMap['filter'] = 'filter'
        colMap['fiveSigmaDepth'] = 'fiveSigmaDepth'
        colMap['night'] = 'night'
        colMap['slewtime'] = 'slewTime'
        colMap['slewdist'] = 'slewDistance'
        colMap['seeingEff'] = 'seeingFwhmEff'
        colMap['seeingGeom'] = 'seeingFwhmGeom'
        colMap['skyBrightness'] = 'skyBrightness'
        colMap['moonDistance'] = 'moonDistance'
        # slew speeds table
        colMap['slewSpeedsTable'] = 'SlewMaxSpeeds'
        colMap['Dome Alt Speed'] = 'domeAltSpeed'
        colMap['Dome Az Speed'] = 'domeAzSpeed'
        colMap['Tel Alt Speed'] = 'telAltSpeed'
        colMap['Tel Az Speed'] = 'telAzSpeed'
        colMap['Rotator Speed'] = 'rotatorSpeed'
        # slew states table
        colMap['slewStatesTable'] = 'SlewFinalStates'
        colMap['Tel Alt'] = 'telAlt'
        colMap['Tel Az'] = 'telAz'
        colMap['Rot Tel Pos'] = 'rotTelPos'
        # slew activities list
        colMap['slewActivitiesTable'] = 'SlewActivities'
        colMap['TelOptics CL'] = 'telopticsclosedloop'
        colMap['TelOptics OL'] = 'telopticsopenloop'
        colMap['Tel Alt'] = 'telalt'
        colMap['Tel Az'] = 'telaz'
        colMap['Tel Settle'] = 'telsettle'
        colMap['Readout'] = 'readout'
        colMap['Dome Alt'] = 'domalt'
        colMap['Dome Az'] = 'domaz'
        colMap['Dome Settle'] = 'domazsettle'
        colMap['Tel Rot'] = 'telrot'
        colMap['Filter'] = 'filter'
        colMap['slewactivities'] = ['Dome Alt', 'Dome Az', 'Dome Settle',
                                    'Tel Alt', 'Tel Az', 'Tel Rot', 'Tel Settle',
                                    'TelOptics CL', 'TelOptics OL',
                                    'Filter', 'Readout']
        colMap['metadataList'] = ['airmass', 'normairmass', 'seeingEff', 'skyBrightness',
                                  'fiveSigmaDepth', 'HA', 'moonDistance', 'solarElong', 'rotSkyPos']

    elif dictName == 'opsimv3':
        colMap = {}
        colMap['ra'] = 'fieldRA'
        colMap['dec'] = 'fieldDec'
        colMap['raDecDeg'] = False
        colMap['mjd'] = 'expMJD'
        colMap['exptime'] = 'visitExpTime'
        colMap['visittime'] = 'visitTime'
        colMap['alt'] = 'altitude'
        colMap['az'] = 'azimuth'
        colMap['filter'] = 'filter'
        colMap['fiveSigmaDepth'] = 'fiveSigmaDepth'
        colMap['night'] = 'night'
        colMap['slewtime'] = 'slewTime'
        colMap['slewdist'] = 'slewDist'
        colMap['seeingEff'] = 'FWHMeff'
        colMap['seeingGeom'] = 'FWHMgeom'
        colMap['skyBrightness'] = 'filtSkyBrightness'
        colMap['moonDistance'] = 'dist2Moon'
        # slew speeds table
        colMap['slewSpeedsTable'] = 'SlewMaxSpeeds'
        colMap['Dome Alt Speed'] = 'domeAltSpeed'
        colMap['Dome Az Speed'] = 'domeAzSpeed'
        colMap['Tel Alt Speed'] = 'telAltSpeed'
        colMap['Tel Az Speed'] = 'telAzSpeed'
        colMap['Rotator Speed'] = 'rotatorSpeed'
        # slew states table
        colMap['slewStatesTable'] = 'SlewStates'
        colMap['Tel Alt'] = 'telAlt'
        colMap['Tel Az'] = 'telAz'
        colMap['Rot Tel Pos'] = 'rotTelPos'
        # Slew activities list
        colMap['slewActivitiesTable'] = 'SlewActivities'
        colMap['TelOptics CL'] = 'TelOpticsOL'
        colMap['TelOptics OL'] = 'telopticsopenloop'
        colMap['Tel Alt'] = 'TelAlt'
        colMap['Tel Az'] = 'TelAz'
        colMap['Readout'] = 'Readout'
        colMap['Dome Alt'] = 'DomAlt'
        colMap['Dome Az'] = 'DomAz'
        colMap['Settle'] = 'Settle'
        colMap['Tel Rot'] = 'Rotator'
        colMap['Filter'] = 'Filter'
        colMap['slewactivities'] = ['Dome Alt', 'Dome Az',
                                    'Tel Alt', 'Tel Az', 'Tel Rot', 'Settle',
                                    'TelOptics CL', 'TelOptics OL',
                                    'Filter', 'Readout']
        colMap['metadataList'] = ['airmass', 'normairmass', 'seeingEff', 'skyBrightness',
                                  'fiveSigmaDepth', 'HA', 'moonDistance', 'solarElong', 'rotSkyPos']

    elif dictName == 'barebones':
        colMap = {}
        colMap['ra'] = 'ra'
        colMap['dec'] = 'dec'
        colMap['raDecDeg'] = True
        colMap['mjd'] = 'mjd'
        colMap['exptime'] = 'exptime'
        colMap['visittime'] = 'exptime'
        colMap['alt'] = 'alt'
        colMap['az'] = 'az'
        colMap['filter'] = 'filter'
        colMap['fiveSigmaDepth'] = 'fivesigmadepth'
        colMap['night'] = 'night'
        colMap['slewtime'] = 'slewtime'
        colMap['slewdist'] = None
        colMap['seeingGeom'] = 'seeing'
        colMap['seeingEff'] = 'seeing'
        colMap['metadataList'] = ['airmass', 'normairmass', 'seeingEff', 'skyBrightness',
                                  'fiveSigmaDepth', 'HA', 'rotSkyPos']

    else:
        raise ValueError('No built in column dict with that name.')

    return colMap
