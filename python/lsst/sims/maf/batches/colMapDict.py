__all__ = ['ColMapDict']


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
        colMap['slewdist'] = 'slewDist'
        colMap['seeingEff'] = 'seeingFwhmEff'
        colMap['seeingGeom'] = 'seeingFwhmGeom'
        # slew speeds table
        colMap['Dome Alt Speed'] = 'domeAltSpeed'
        colMap['Dome Az Speed'] = 'domeAzSpeed'
        colMap['Tel Alt Speed'] = 'telAltSpeed'
        colMap['Tel Az Speed'] = 'telAzSpeed'
        colMap['Rotator Speed'] = 'rotatorSpeed'
        # slew states table
        colMap['Tel Alt'] = 'telAlt'
        colMap['Tel Az'] = 'telAz'
        colMap['Rot Tel Pos'] = 'rotTelPos'
        # slew activities list
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
        colMap['slewdist'] = 'slewDistance'
        colMap['seeingEff'] = 'FWHMeff'
        colMap['seeingGeom'] = 'FWHMgeom'
        # slew speeds table
        colMap['Dome Alt Speed'] = 'domeAltSpeed'
        colMap['Dome Az Speed'] = 'domeAzSpeed'
        colMap['Tel Alt Speed'] = 'telAltSpeed'
        colMap['Tel Az Speed'] = 'telAzSpeed'
        colMap['Rotator Speed'] = 'rotatorSpeed'
        # slew states table
        colMap['Tel Alt'] = 'telAlt'
        colMap['Tel Az'] = 'telAz'
        colMap['Rot Tel Pos'] = 'rotTelPos'
        # Slew activities list
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

    else:
        raise ValueError('No built in column dict with that name.')

    return colMap