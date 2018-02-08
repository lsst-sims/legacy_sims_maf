from builtins import zip
from builtins import range
import numpy as np
from .baseStacker import BaseStacker

__all__ = ['wrapRADec', 'wrapRA', 'inHexagon', 'polygonCoords',
           'RandomDitherFieldPerVisitStacker', 'RandomDitherFieldPerNightStacker',
           'RandomDitherPerNightStacker',
           'SpiralDitherFieldPerVisitStacker', 'SpiralDitherFieldPerNightStacker',
           'SpiralDitherPerNightStacker',
           'HexDitherFieldPerVisitStacker', 'HexDitherFieldPerNightStacker',
           'HexDitherPerNightStacker', 'DefaultDitherStacker']

# Stacker naming scheme:
# [Pattern]Dither[Field]Per[Timescale].
#  Timescale indicates how often the dither offset is changed.
#  The presence of 'Field' indicates that a new offset is chosen per field, on the indicated timescale.
#  The absence of 'Field' indicates that all visits within the indicated timescale use the same dither offset.


# Original dither stackers (Random, Spiral, Hex) written by Lynne Jones (lynnej@uw.edu)
# Additional dither stackers written by Humna Awan (humna.awan@rutgers.edu), with addition of
# constraining dither offsets to be within an inscribed hexagon (code modifications for use here by LJ).


def wrapRADec(ra, dec):
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians

    Returns
    -------
    numpy.ndarray
        Wrapped RA values, in radians.
    """
    ra = ra % (2.0 * np.pi)
    return ra


def inHexagon(xOff, yOff, maxDither):
    """
    Identify dither offsets which fall within the inscribed hexagon.

    Parameters
    ----------
    xOff : numpy.ndarray
        The x values of the dither offsets.
    yoff : numpy.ndarray
        The y values of the dither offsets.
    maxDither : float
        The maximum dither offset.

    Returns
    -------
    numpy.ndarray
        Indexes of the offsets which are within the hexagon inscribed inside the 'maxDither' radius circle.
    """
    # Set up the hexagon limits.
    #  y = mx + b, 2h is the height.
    m = np.sqrt(3.0)
    b = m * maxDither
    h = m / 2.0 * maxDither
    # Identify offsets inside hexagon.
    inside = np.where((yOff < m * xOff + b) &
                      (yOff > m * xOff - b) &
                      (yOff < -m * xOff + b) &
                      (yOff > -m * xOff - b) &
                      (yOff < h) & (yOff > -h))[0]
    return inside


def polygonCoords(nside, radius, rotationAngle):
    """
    Find the x,y coords of a polygon.

    This is useful for plotting dither points and showing they lie within
    a given shape.

    Parameters
    ----------
    nside : int
        The number of sides of the polygon
    radius : float
        The radius within which to plot the polygon
    rotationAngle : float
        The angle to rotate the polygon to.

    Returns
    -------
    [float, float]
        List of x/y coordinates of the points describing the polygon.
    """
    eachAngle = 2 * np.pi / float(nside)
    xCoords = np.zeros(nside, float)
    yCoords = np.zeros(nside, float)
    for i in range(0, nside):
        xCoords[i] = np.sin(eachAngle * i + rotationAngle) * radius
        yCoords[i] = np.cos(eachAngle * i + rotationAngle) * radius
    return list(zip(xCoords, yCoords))


class RandomDitherFieldPerVisitStacker(BaseStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    with a different offset for each field, for each visit.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, maxDither=1.75,
                 inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        # Convert maxDither to radians for internal use.
        self.maxDither = np.radians(maxDither)
        self.inHex = inHex
        self.randomSeed = randomSeed
        # self.units used for plot labels
        if self.degrees:
            self.units = ['deg', 'deg']
        else:
            self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['randomDitherFieldPerVisitRa', 'randomDitherFieldPerVisitDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol]

    def _generateRandomOffsets(self, noffsets):
        xOut = np.array([], float)
        yOut = np.array([], float)
        maxTries = 100
        tries = 0
        while (len(xOut) < noffsets) and (tries < maxTries):
            dithersRad = np.sqrt(np.random.rand(noffsets * 2)) * self.maxDither
            dithersTheta = np.random.rand(noffsets * 2) * np.pi * 2.0
            xOff = dithersRad * np.cos(dithersTheta)
            yOff = dithersRad * np.sin(dithersTheta)
            if self.inHex:
                # Constrain dither offsets to be within hexagon.
                idx = inHexagon(xOff, yOff, self.maxDither)
                xOff = xOff[idx]
                yOff = yOff[idx]
            xOut = np.concatenate([xOut, xOff])
            yOut = np.concatenate([yOut, yOff])
            tries += 1
        if len(xOut) < noffsets:
            raise ValueError('Could not find enough random points within the hexagon in %d tries. '
                             'Try another random seed?' % (maxTries))
        self.xOff = xOut[0:noffsets]
        self.yOff = yOut[0:noffsets]

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Generate the random dither values.
        noffsets = len(simData[self.raCol])
        self._generateRandomOffsets(noffsets)
        # Add to RA and dec values.
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        simData['randomDitherFieldPerVisitRa'] = (ra + self.xOff / np.cos(dec))
        simData['randomDitherFieldPerVisitDec'] = dec + self.yOff
        # Wrap back into expected range.
        simData['randomDitherFieldPerVisitRa'], simData['randomDitherFieldPerVisitDec'] = \
            wrapRADec(simData['randomDitherFieldPerVisitRa'], simData['randomDitherFieldPerVisitDec'])
        # Convert to degrees
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class RandomDitherFieldPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    one dither offset per new night of observation of a field.
    e.g. visits within the same night, to the same field, have the same offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId', nightCol='night',
                 maxDither=1.75, inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super(RandomDitherFieldPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                               maxDither=maxDither, inHex=inHex,
                                                               randomSeed=randomSeed)
        self.nightCol = nightCol
        self.fieldIdCol = fieldIdCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['randomDitherFieldPerNightRa', 'randomDitherFieldPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol, self.fieldIdCol]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Generate the random dither values, one per night per field.
        fields = np.unique(simData[self.fieldIdCol])
        nights = np.unique(simData[self.nightCol])
        self._generateRandomOffsets(len(fields) * len(nights))
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # counter to ensure new random numbers are chosen every time
        delta = 0
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply dithers, increasing each night.
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % len(self.xOff)
            # ensure that the same xOff/yOff entries are not chosen
            delta = delta + len(vertexIdxs)
            simData['randomDitherFieldPerNightRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['randomDitherFieldPerNightDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['randomDitherFieldPerNightRa'], simData['randomDitherFieldPerNightDec'] = \
            wrapRADec(simData['randomDitherFieldPerNightRa'], simData['randomDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class RandomDitherPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    one dither offset per night.
    All fields observed within the same night get the same offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, nightCol='night',
                 maxDither=1.75, inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super(RandomDitherPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                          maxDither=maxDither, inHex=inHex,
                                                          randomSeed=randomSeed)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['randomDitherPerNightRa', 'randomDitherPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Generate the random dither values, one per night.
        nights = np.unique(simData[self.nightCol])
        self._generateRandomOffsets(len(nights))
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        for n, x, y in zip(nights, self.xOff, self.yOff):
            match = np.where(simData[self.nightCol] == n)[0]
            simData['randomDitherPerNightRa'][match] = (ra[match] +
                                                        x / np.cos(dec[match]))
            simData['randomDitherPerNightDec'][match] = dec[match] + y
        # Wrap RA/Dec into expected range.
        simData['randomDitherPerNightRa'], simData['randomDitherPerNightDec'] = \
            wrapRADec(simData['randomDitherPerNightRa'], simData['randomDitherPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherFieldPerVisitStacker(BaseStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Each visit to a field receives a new, sequential offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        self.fieldIdCol = fieldIdCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.numPoints = numPoints
        self.nCoils = nCoils
        self.maxDither = np.radians(maxDither)
        self.inHex = inHex
        # self.units used for plot labels
        if self.degrees:
            self.units =  ['deg', 'deg']
        else:
            self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['spiralDitherFieldPerVisitRa', 'spiralDitherFieldPerVisitDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateSpiralOffsets(self):
        # First generate a full archimedean spiral ..
        theta = np.arange(0.0001, self.nCoils * np.pi * 2., 0.001)
        a = self.maxDither/theta.max()
        if self.inHex:
            a = 0.85 * a
        r = theta * a
        # Then pick out equidistant points along the spiral.
        arc = a / 2.0 * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))
        stepsize = arc.max()/float(self.numPoints)
        arcpts = np.arange(0, arc.max(), stepsize)
        arcpts = arcpts[0:self.numPoints]
        rpts = np.zeros(self.numPoints, float)
        thetapts = np.zeros(self.numPoints, float)
        for i, ap in enumerate(arcpts):
            diff = np.abs(arc - ap)
            match = np.where(diff == diff.min())[0]
            rpts[i] = r[match]
            thetapts[i] = theta[match]
        # Translate these r/theta points into x/y (ra/dec) offsets.
        self.xOff = rpts * np.cos(thetapts)
        self.yOff = rpts * np.sin(thetapts)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate the spiral offset vertices.
        self._generateSpiralOffsets()
        # Now apply to observations.
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['spiralDitherFieldPerVisitRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['spiralDitherFieldPerVisitDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['spiralDitherFieldPerVisitRa'], simData['spiralDitherFieldPerVisitDec'] = \
            wrapRADec(simData['spiralDitherFieldPerVisitRa'], simData['spiralDitherFieldPerVisitDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherFieldPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Each field steps along a sequential series of offsets, each night it is observed.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId', nightCol='night',
                 numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        super(SpiralDitherFieldPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                               fieldIdCol=fieldIdCol,
                                                               numPoints=numPoints, maxDither=maxDither,
                                                               nCoils=nCoils, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['spiralDitherFieldPerNightRa', 'spiralDitherFieldPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateSpiralOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['spiralDitherFieldPerNightRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['spiralDitherFieldPerNightDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['spiralDitherFieldPerNightRa'], simData['spiralDitherFieldPerNightDec'] = \
            wrapRADec(simData['spiralDitherFieldPerNightRa'], simData['spiralDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    All fields observed in the same night receive the same sequential offset, changing per night.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId', nightCol='night',
                 numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        super(SpiralDitherPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                          fieldIdCol=fieldIdCol,
                                                          numPoints=numPoints, maxDither=maxDither,
                                                          nCoils=nCoils, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['spiralDitherPerNightRa', 'spiralDitherPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateSpiralOffsets()
        nights = np.unique(simData[self.nightCol])
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        vertexIdxs = np.searchsorted(nights, simData[self.nightCol])
        vertexIdxs = vertexIdxs % self.numPoints
        simData['spiralDitherPerNightRa'] = (ra +
                                             self.xOff[vertexIdxs] / np.cos(dec))
        simData['spiralDitherPerNightDec'] = dec + self.yOff[vertexIdxs]
        # Wrap RA/Dec into expected range.
        simData['spiralDitherPerNightRa'], simData['spiralDitherPerNightDec'] = \
            wrapRADec(simData['spiralDitherPerNightRa'], simData['spiralDitherPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherFieldPerVisitStacker(BaseStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each visit.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True,
                 fieldIdCol='fieldId', maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        self.fieldIdCol = fieldIdCol
        self.maxDither = np.radians(maxDither)
        self.inHex = inHex
        # self.units used for plot labels
        if self.degrees:
            self.units = ['deg', 'deg']
        else:
            self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['hexDitherFieldPerVisitRa', 'hexDitherFieldPerVisitDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateHexOffsets(self):
        # Set up basics of dither pattern.
        dith_level = 4
        nrows = 2**dith_level
        halfrows = int(nrows / 2.)
        # Calculate size of each offset
        dith_size_x = self.maxDither * 2.0 / float(nrows)
        dith_size_y = np.sqrt(3) * self.maxDither / float(nrows)  # sqrt 3 comes from hexagon
        if self.inHex:
            dith_size_x = 0.95 * dith_size_x
            dith_size_y = 0.95 * dith_size_y
        # Calculate the row identification number, going from 0 at center
        nid_row = np.arange(-halfrows, halfrows + 1, 1)
        # and calculate the number of vertices in each row.
        vert_in_row = np.arange(-halfrows, halfrows + 1, 1)
        # First calculate how many vertices we will create in each row.
        total_vert = 0
        for i in range(-halfrows, halfrows + 1, 1):
            vert_in_row[i] = (nrows+1) - abs(nid_row[i])
            total_vert += vert_in_row[i]
        self.numPoints = total_vert
        self.xOff = []
        self.yOff = []
        # Calculate offsets over hexagonal grid.
        for i in range(0, nrows+1, 1):
            for j in range(0, vert_in_row[i], 1):
                self.xOff.append(dith_size_x * (j - (vert_in_row[i] - 1) / 2.0))
                self.yOff.append(dith_size_y * nid_row[i])
        self.xOff = np.array(self.xOff)
        self.yOff = np.array(self.yOff)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateHexOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['hexDitherFieldPerVisitRa'][match] = (ra[match] +
                                                          self.xOff[vertexIdxs] /
                                                          np.cos(dec[match]))
            simData['hexDitherFieldPerVisitDec'][match] = dec[match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['hexDitherFieldPerVisitRa'], simData['hexDitherFieldPerVisitDec'] = \
            wrapRADec(simData['hexDitherFieldPerVisitRa'], simData['hexDitherFieldPerVisitDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherFieldPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each night of visits.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True,
                 fieldIdCol='fieldId', nightCol='night',
                 maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        super(HexDitherFieldPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, fieldIdCol=fieldIdCol,
                                                            degrees=degrees, maxDither=maxDither, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['hexDitherFieldPerNightRa', 'hexDitherFieldPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateHexOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['hexDitherFieldPerNightRa'][match] = (ra[match] +
                                                          self.xOff[vertexIdxs] /
                                                          np.cos(dec[match]))
            simData['hexDitherFieldPerNightDec'][match] = (dec[match] +
                                                           self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['hexDitherFieldPerNightRa'], simData['hexDitherFieldPerNightDec'] = \
            wrapRADec(simData['hexDitherFieldPerNightRa'], simData['hexDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset per night for all fields.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        super(HexDitherPerNightStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                       fieldIdCol=fieldIdCol,
                                                       maxDither=maxDither, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)
        self.addedRA = 'hexDitherPerNightRa'
        self.addedDec = 'hexDitherPerNightDec'
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = [self.addedRA, self.addedDec]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate the spiral dither values
        self._generateHexOffsets()
        nights = np.unique(simData[self.nightCol])
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        vertexID = 0
        for n in nights:
            match = np.where(simData[self.nightCol] == n)[0]
            vertexID = vertexID % self.numPoints
            simData[self.addedRA][match] = (ra[match] + self.xOff[vertexID] / np.cos(dec[match]))
            simData[self.addedDec][match] = dec[match] + self.yOff[vertexID]
            vertexID += 1
        # Wrap RA/Dec into expected range.
        simData[self.addedRA], simData[self.addedDec] = \
            wrapRADec(simData[self.addedRA], simData[self.addedDec])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class DefaultDitherStacker(HexDitherPerNightStacker):
    """
    Make a default dither pattern to stack on ditheredRA and ditheredDec.
    The default pattern is HexDitherPerNightStacker.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', maxDither=1.75, inHex=True):
        super(DefaultDitherStacker, self).__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                                                   fieldIdCol=fieldIdCol, nightCol=nightCol,
                                                   maxDither=maxDither, inHex=inHex)
        self.addedRA = 'ditheredRA'
        self.addedDec = 'ditheredDec'
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = [self.addedRA, self.addedDec]


