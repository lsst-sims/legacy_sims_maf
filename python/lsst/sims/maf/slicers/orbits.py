import warnings
import numpy as np
import pandas as pd

__all__ = ['Orbits']


class Orbits(object):
    """Orbits reads and stores orbit parameters for moving objects.
    """
    def __init__(self):
        self.orbits = None
        self.format = None

        # Specify the required columns/values in the self.orbits dataframe.
        # Which columns are required depends on self.format.
        self.dataCols = {}
        self.dataCols['COM'] = ['objId', 'q', 'e', 'inc', 'Omega', 'argPeri',
                                'tPeri', 'epoch', 'H', 'g', 'sed_filename']
        self.dataCols['KEP'] = ['objId', 'a', 'e', 'inc', 'Omega', 'argPeri',
                                'meanAnomaly', 'epoch', 'H', 'g', 'sed_filename']

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, i):
        orb = Orbits()
        orb.setOrbits(self.orbits.iloc[i])
        return orb

    def __iter__(self):
        for i, orbit in self.orbits.iterrows():
            orb = Orbits()
            orb.setOrbits(orbit)
            yield orb

    def __eq__(self, otherOrbits):
        if isinstance(otherOrbits, Orbits):
            if self.format != otherOrbits.format:
                return False
            for col in self.dataCols[self.format]:
                if not self.orbits[col].equals(otherOrbits.orbits[col]):
                    return False
                else:
                    return True
        else:
            return False

    def __neq__(self, otherOrbits):
        if self == otherOrbits:
            return False
        else:
            return True

    def setOrbits(self, orbits):
        """Set and validate orbital parameters contain all required values.

        Sets self.orbits and self.format.
        If objid is not present in orbits, a sequential series of integers will be used.
        If H is not present in orbits, a default value of 20 will be used.
        If g is not present in orbits, a default value of 0.15 will be used.
        If sed_filename is not present in orbits, either C or S type will be assigned,
        according to the semi-major axis value.

        Parameters
        ----------
        orbits : pandas.DataFrame, pandas.Series or numpy.ndarray
           Array-like object containing orbital parameter information.
        """
        # Do we have a single item or multiples?
        if isinstance(orbits, pd.Series):
            # Passed a single SSO in Series, convert to a DataFrame.
            orbits = pd.DataFrame([orbits])
        elif isinstance(orbits, np.ndarray):
            # Passed a numpy array, convert to DataFrame.
            orbits = pd.DataFrame.from_records(orbits)
        elif isinstance(orbits, np.record):
            # This was a single object in a numpy array and we should be a bit fancy.
            orbits = pd.DataFrame.from_records([orbits], columns=orbits.dtype.names)

        if 'index' in orbits:
            del orbits['index']

        nSso = len(orbits)

        # Error if orbits is empty (this avoids hard-to-interpret error messages from pyoorb).
        if nSso == 0:
            raise ValueError('Length of the orbits dataframe was 0.')

        # Discover which type of orbital parameters we have on disk.
        format = None
        if 'FORMAT' in orbits:
            format = orbits['FORMAT'].iloc[0]
            del orbits['FORMAT']
        if 'q' in orbits:
            self.format = 'COM'
        elif 'a' in orbits:
            self.format = 'KEP'
        else:
            raise ValueError('Cannot determine orbital type, as neither q nor a in input orbital elements.\n'
                             'Was attempting to base orbital element quantities on header row, '
                             'with columns: \n%s' % orbits.columns)
        # Report a warning if formats don't seem to match.
        if (format is not None) and (format != self.format):
            warnings.warn("Format from input file (%s) doesn't match determined format (%s). "
                          "Using %s" % (format, self.format, self.format))

        # Check that the orbit epoch is within a 'reasonable' range, to detect possible column mismatches.
        general_epoch = orbits['epoch'].head(1).values[0]
        expect_min_epoch = 16000.
        expect_max_epoch = 80000.
        if general_epoch < expect_min_epoch or general_epoch > expect_max_epoch:
            raise ValueError("The epoch detected for this orbit is odd - %f. "
                             "Expecting a value between %.1f and %.1f" % (general_epoch,
                                                                          expect_min_epoch,
                                                                          expect_max_epoch))

        # If these columns are not available in the input data, auto-generate them.
        if 'objId' not in orbits:
            orbits['objId'] = np.arange(0, nSso, 1)
        if 'H' not in orbits:
            orbits['H'] = np.zeros(nSso) + 20.0
        if 'g' not in orbits:
            orbits['g'] = np.zeros(nSso) + 0.15
        if 'sed_filename' not in orbits:
            orbits['sed_filename'] = self.assignSed(orbits)

        # Make sure we gave all the columns we need.
        for col in self.dataCols[self.format]:
            if col not in orbits:
                raise ValueError('Missing required orbital element %s for orbital format type %s'
                                 % (col, self.format))

        # Check to see if we have duplicates.
        if len(np.unique(orbits['objId'])) != nSso:
            warnings.warn('There are duplicates in the orbit objId values' +
                          ' - was this intended? (continuing).')
        # All is good.
        self.orbits = orbits

    def assignSed(self, orbits, randomSeed=None):
        """Assign either a C or S type SED, depending on the semi-major axis of the object.
        P(C type) = 0 (a<2); 0.5*a - 1 (2<a<4); 1 (a > 4),
        based on figure 23 from Ivezic et al 2001 (AJ, 122, 2749).

        Parameters
        ----------
        orbits : pandas.DataFrame, pandas.Series or numpy.ndarray
           Array-like object containing orbital parameter information.

        Returns
        -------
        numpy.ndarray
            Array containing the SED type for each object in 'orbits'.
        """
        # using fig. 23 from Ivezic et al. 2001 (AJ, 122, 2749),
        # we can approximate the sed types with a simple linear form:
        #  p(C) = 0 for a<2
        #  p(C) = 0.5*a-1  for 2<a<4
        #  p(C) = 1 for a>4
        # where a is semi-major axis, and p(C) is the probability that
        # an asteroid is C type, with p(S)=1-p(C) for S types.
        if 'a' in orbits:
            a = orbits['a']
        elif 'q' in orbits:
            a = orbits['q'] / (1 - orbits['e'])
        else:
            raise ValueError('Need either a or q (plus e) in orbit data frame.')
        sedvals = np.empty(len(orbits), dtype=str)
        if randomSeed is not None:
            np.random.seed(randomSeed)
        chance = np.random.random(len(orbits))
        prob_c = 0.5 * a - 1.0
        # if chance <= prob_c:
        sedvals = np.where(chance <= prob_c, 'C.dat', 'S.dat')
        return sedvals

    def readOrbits(self, orbitfile, delim=None, skiprows=None):
        """Read orbits from a file, generating a pandas dataframe containing columns matching
        dataCols, for the appropriate orbital parameter format (currently accepts COM or KEP formats).

        After reading and standardizing the column names, calls selfs.setOrbits to validate the
        orbital parameters. Expects angles in orbital element formats to be in degrees.

        Parameters
        ----------
        orbitfile : str
            The name of the input file containing orbital parameter information.
        delim : str, optional
            The delimiter for the input orbit file -- default = None will use delim_whitespace=True.
        skiprows : int, optional
            The number of rows to skip before reading the header information for pandas.
        """
        names = None
        if skiprows is None:
            skiprows = 0
            # Figure out whether the header is in the first line, or if there are rows to skip.
            # We need to do a bit of juggling to do this before pandas reads the whole orbit file though.
            file = open(orbitfile, 'r')
            for line in file:
                values = line.split()
                try:
                    # If it is a valid orbit line, we expect 3 to be eccentricity.
                    float(values[3])
                    # And if it worked, we're done here.
                    break
                except (ValueError, IndexError):
                    # This wasn't a valid number or there wasn't anything in the third value
                    skiprows += 1
                    valuesheader = values
            skiprows -= 1
            file.close()

        if skiprows == -1:
            # No header; assume it's a typical DES file.
            names = ('objId', 'FORMAT', 'q', 'e', 'i', 'node', 'argperi', 't_p',
                     'H',  'epoch', 'INDEX', 'N_PAR', 'MOID', 'COMPCODE')
            orbits = pd.read_table(orbitfile, delim_whitespace=True, skiprows=0,
                                   names=names)

        else:
            # There is a header, but we also need to check if there is a comment key at the start
            # of the proper header line.
            linestart = valuesheader[0]
            if linestart == '#' or linestart == '!!' or linestart == '##':
                names = valuesheader[1:]
                skiprows += 1
            # Read the data from disk.
            if delim is None:
                orbits = pd.read_table(orbitfile, delim_whitespace=True, names=names, skiprows=skiprows)
            else:
                orbits = pd.read_table(orbitfile, sep=delim, names=names, skiprows=skiprows)

        # Drop some columns that are typically present in DES files but that we don't need.
        if 'INDEX' in orbits:
            del orbits['INDEX']
        if 'N_PAR' in orbits:
            del orbits['N_PAR']
        if 'MOID' in orbits:
            del orbits['MOID']
        if 'COMPCODE' in orbits:
            del orbits['COMPCODE']
        if 'tmp' in orbits:
            del orbits['tmp']

        # Normalize the column names to standard values and identify the orbital element types.
        ssoCols = orbits.columns.values.tolist()

        # These are the alternative possibilities for various column headers
        # (depending on file version, origin, etc.)
        # that might need remapping from the on-file values to our standardized values.
        altNames = {}
        altNames['objId'] = ['objId', 'objid', '!!ObjID', '!!OID', '!!S3MID', 'OID', 'S3MID'
                             'objid(int)', 'full_name', '#name']
        altNames['q'] = ['q']
        altNames['a'] = ['a']
        altNames['e'] = ['e', 'ecc']
        altNames['inc'] = ['inc', 'i', 'i(deg)', 'incl']
        altNames['Omega'] = ['Omega', 'omega', 'node', 'om', 'node(deg)',
                             'BigOmega', 'Omega/node', 'longNode']
        altNames['argPeri'] = ['argPeri', 'argperi', 'omega/argperi', 'w', 'argperi(deg)', 'peri']
        altNames['tPeri'] = ['tPeri', 't_p', 'timeperi', 't_peri', 'T_peri']
        altNames['epoch'] = ['epoch', 't_0', 'Epoch', 'epoch_mjd']
        altNames['H'] = ['H', 'magH', 'magHv', 'Hv', 'H_v']
        altNames['g'] = ['g', 'phaseV', 'phase', 'gV', 'phase_g', 'G']
        altNames['meanAnomaly'] = ['meanAnomaly', 'meanAnom', 'M', 'ma']
        altNames['sed_filename'] = ['sed_filename', 'sed']

        # Update column names that match any of the alternatives above.
        for name, alternatives in altNames.items():
            intersection = list(set(alternatives) & set(ssoCols))
            if len(intersection) > 1:
                raise ValueError('Received too many possible matches to %s in orbit file %s'
                                 % (name, orbitfile))
            if len(intersection) == 1:
                idx = ssoCols.index(intersection[0])
                ssoCols[idx] = name
        # Assign the new column names back to the orbits dataframe.
        orbits.columns = ssoCols
        # Validate and assign orbits to self.
        self.setOrbits(orbits)
