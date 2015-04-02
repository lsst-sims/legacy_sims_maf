import numpy as np
from .baseStacker import BaseStacker

__all__ = ['wrapRADec', 'wrapRA',
           'RandomDitherStacker', 'NightlyRandomDitherStacker',
           'SpiralDitherStacker', 'NightlySpiralDitherStacker',
           'SequentialHexDitherStacker', 'NightlySequentialHexDitherStacker']

def wrapRADec(ra, dec):
    """
    Wrap RA and Dec values so RA between 0-2pi (using mod),
      and Dec in +/- pi/2.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi/2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi/2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0*np.pi)
    return ra, dec

def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).
    """
    ra = ra % (2.0*np.pi)
    return ra

class RandomDitherStacker(BaseStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center, different offset per visit.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', maxDither=1.75, randomSeed=None):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = np.radians(maxDither)
        self.randomSeed = randomSeed
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['randomRADither', 'randomDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol]

    def _generateRandomOffsets(self, noffsets):
        dithersRad = np.sqrt(np.random.rand(noffsets))*self.maxDither
        dithersTheta = np.random.rand(noffsets)*np.pi*2.0
        self.xOff = dithersRad * np.cos(dithersTheta)
        self.yOff = dithersRad * np.sin(dithersTheta)

    def run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Add new columns to simData, ready to fill with new values.
        simData = self._addStackers(simData)
        # Generate the random dither values.
        noffsets = len(simData[self.raCol])
        self._generateRandomOffsets(noffsets)
        # Add to RA and dec values.
        simData['randomRADither'] = simData[self.raCol] + self.xOff/np.cos(simData[self.decCol])
        simData['randomDecDither'] = simData[self.decCol] + self.yOff
        # Wrap back into expected range.
        simData['randomRADither'], simData['randomDecDither'] = wrapRADec(simData['randomRADither'], simData['randomDecDither'])
        return simData

class NightlyRandomDitherStacker(RandomDitherStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center, one dither offset per night.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night', maxDither=1.75, randomSeed=None):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        self.nightCol = nightCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = np.radians(maxDither)
        self.randomSeed = randomSeed
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['nightlyRandomRADither', 'nightlyRandomDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol]

    def run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
        # Add the new columns to simData.
        simData = self._addStackers(simData)
        # Generate the random dither values, one per night.
        nights = np.unique(simData[self.nightCol])
        self._generateRandomOffsets(len(nights))
        # Add to RA and dec values.
        for n, x, y in zip(nights, self.xOff, self.yOff):
            match = np.where(simData[self.nightCol] == n)[0]
            simData['nightlyRandomRADither'][match] = simData[self.raCol][match] + x/np.cos(simData[self.decCol][match])
            simData['nightlyRandomDecDither'][match] = simData[self.decCol][match] + y
        # Wrap RA/Dec into expected range.
        simData['nightlyRandomRADither'], simData['nightlyRandomDecDither'] = wrapRADec(simData['nightlyRandomRADither'],
                                                                                        simData['nightlyRandomDecDither'])
        return simData


class SpiralDitherStacker(BaseStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Sequential offset for each individual visit to a field.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID',
                 numPoints=60, maxDither=1.75, nCoils=5):
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol = fieldIdCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.numPoints = numPoints
        self.nCoils = nCoils
        self.maxDither = np.radians(maxDither)
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['spiralRADither', 'spiralDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateSpiralOffsets(self):
        # First generate a full archimedean spiral ..
        theta = np.arange(0.0001, self.nCoils*np.pi*2., 0.001)
        a = self.maxDither/theta.max()
        r = theta*a
        # Then pick out equidistant points along the spiral.
        arc = a / 2.0 *(theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))
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

    def run(self, simData):
        # Add the new columns to simData.
        simData = self._addStackers(simData)
        # Generate the spiral offset vertices.
        self._generateSpiralOffsets()
        # Now apply to observations.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['spiralRADither'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['spiralDecDither'][match] = simData[self.decCol][match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['spiralRADither'], simData['spiralDecDither'] = wrapRADec(simData['spiralRADither'],
                                                                          simData['spiralDecDither'])
        return simData


class NightlySpiralDitherStacker(SpiralDitherStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Sequential offset for each night of visits to a field.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID', nightCol='night',
                 numPoints=60, maxDither=1.75, nCoils=5):
        super(NightlySpiralDitherStacker, self).__init__(raCol=raCol, decCol=decCol, fieldIdCol=fieldIdCol,
                                                         numPoints=numPoints, maxDither=maxDither, nCoils=nCoils)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['nightlySpiralRADither', 'nightlySpiralDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def run(self, simData):
        simData = self._addStackers(simData)
        self._generateSpiralOffsets()
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['nightlySpiralRADither'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['nightlySpiralDecDither'][match] = simData[self.decCol][match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['nightlySpiralRADither'], simData['nightlySpiralDecDither'] = wrapRADec(simData['nightlySpiralRADither'],
                                                                                        simData['nightlySpiralDecDither'])
        return simData


class SequentialHexDitherStacker(BaseStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each visit.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID', maxDither=1.8):
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol = fieldIdCol
        self.maxDither = np.radians(maxDither)
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['sequentialHexRADither', 'sequentialHexDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol]

    def _generateHexOffsets(self):
        # Set up basics of dither pattern.
        dith_level = 4
        nrows = 2**dith_level
        halfrows = int(nrows/2.)
        # Calculate size of each offset
        dith_size_x = self.maxDither*2.0/float(nrows)
        dith_size_y = np.sqrt(3)*self.maxDither/float(nrows)  #sqrt 3 comes from hexagon
        # Calculate the row identification number, going from 0 at center
        nid_row = np.arange(-halfrows, halfrows+1, 1)
        # and calculate the number of vertices in each row.
        vert_in_row = np.arange(-halfrows, halfrows+1, 1)
        # First calculate how many vertices we will create in each row.
        total_vert = 0
        for i in range(-halfrows, halfrows+1, 1):
            vert_in_row[i] = (nrows+1) - abs(nid_row[i])
            total_vert += vert_in_row[i]
        self.numPoints = total_vert
        self.xOff = []
        self.yOff = []
        # Calculate offsets over hexagonal grid.
        for i in range(0, nrows+1, 1):
            for j in range(0, vert_in_row[i], 1):
                self.xOff.append(dith_size_x * (j - (vert_in_row[i]-1)/2.0))
                self.yOff.append(dith_size_y * nid_row[i])
        self.xOff = np.array(self.xOff)
        self.yOff = np.array(self.yOff)

    def run(self, simData):
        simData = self._addStackers(simData)
        self._generateHexOffsets()
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither.
            vertexIdxs = np.arange(0, len(match), 1)
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['sequentialHexRADither'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['sequentialHexDecDither'][match] = simData[self.decCol][match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['sequentialHexRADither'], simData['sequentialHexDecDither'] = wrapRADec(simData['sequentialHexRADither'],
                                                                                        simData['sequentialHexDecDither'])
        return simData

class NightlySequentialHexDitherStacker(SequentialHexDitherStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each night of visits.
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldIdCol', nightCol='night', maxDither=1.8):
        super(NightlySequentialHexDitherStacker, self).__init__(raCol=raCol, decCol=decCol, maxDither=maxDither)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['nightlySequentialHexRADither', 'nightlySequentialHexDecDither']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def run(self, simData):
        simData = self._addStackers(simData)
        self._generateHexOffsets()
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['nightlySequentialHexRADither'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['nightlySequentialHexDecDither'][match] = simData[self.decCol][match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['nightlySequentialHexRADither'], simData['nightlySequentialHexDecDither'] = \
          wrapRADec(simData['nightlySequentialHexRADither'], simData['nightlySequentialHexDecDither'])
        return simData
