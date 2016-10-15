import numpy as np
import pandas as pd
from itertools import repeat

class MoOrbits(object):
    """
    This class stores the orbits for a set of moving objects (and reads them from disk).

    """
    def __init__(self):
        self.orbits = None
        self.ssoIds = None
        self.nSso = 0


    def _updateColMap(self, colMap, outCol, alternativeNames, ssoCols):
        """
        Private method used by readOrbits to update the map between data file and final orbit columns.

        This map is needed because data files can use different column headers for the same quantity.
        """
        for a in alternativeNames:
            if a in ssoCols:
                colMap[outCol] = a
        return colMap

    def readOrbits(self, orbitfile, delim=None):
        """
        Read the orbits from file 'orbitfile', generating a numpy structured array with the columns:
        'objID q e inc node argPeri tPeri epoch H g a meanAnom sed_filename'
        """
        self.orbitfile = orbitfile
        if delim is None:
            orbits = pd.read_table(orbitfile, delim_whitespace=True)
        else:
            orbits = pd.read_table(orbitfile, sep = delim)
        # Normalize the column names, as different inputs tend to have some commonly-different names.
        ssoCols = orbits.columns.values.tolist()
        nSso = len(orbits)
        outCols = ['objId', 'q', 'e', 'inc', 'node', 'argPeri', 'tPeri', 'epoch', 'H', 'g', 'a', 'meanAnom',
                   'sed_filename']
        # Create mapping between column names read from disk and desired column names.
        colMap = {}
        for o in outCols:
            if o in ssoCols:
                colMap[o] = o
            else:
                # Try to find corresponding value
                if o == 'objId':
                    alternatives = ['!!ObjID', 'objid', 'objid(int)', '!!OID', 'full_name']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'inc':
                    alternatives = ['i', 'i(deg)']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'node':
                    alternatives = ['BigOmega', 'Omega/node', 'Omega', 'om', 'node(deg)']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'argPeri':
                    alternatives = ['argperi', 'omega/argperi', 'w', 'argperi(deg)']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'tPeri':
                    alternatives = ['t_p', 'timeperi', 't_peri']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'epoch':
                    alternatives = ['t_0', 'Epoch', 'epoch_mjd']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'H':
                    alternatives = ['magH', 'magHv', 'Hv', 'H_v']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'g':
                    alternatives = ['phaseV', 'phase', 'gV', 'phase_g']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'a':
                    alternatives = ['semimajor']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)
                elif o == 'meanAnom':
                    alternatives = ['M', 'meanAnomaly', 'ma']
                    colMap = self._updateColMap(colMap, o, alternatives, ssoCols)

        # Add the columns we can generate with some guesses.
        if 'objId' not in colMap:
            orbids = np.arange(0, nSso, 1)
        else:
            orbids = orbits[colMap['objId']]
        if 'H' not in colMap:
            Hval = np.zeros(nSso) + 20.0
        else:
            Hval = orbits[colMap['H']]
        if 'g' not in colMap:
            gval = np.zeros(nSso) + 0.15
        else:
            gval = orbits[colMap['g']]
        if 'sed_filename' not in colMap:
            sedvals = [sed for sed in repeat('C.dat', nSso)]
            sedvals = np.array(sedvals)
        else:
            sedvals = orbits[colMap['sed_filename']]

        # And some columns that can be generated from the input data we do have.
        # This is probably not as reliable as it needs to be ..
        # converting from a/M to q/tPeri is not accurate enough.
        if 'a' not in colMap:
            aval = orbits[colMap['q']] / (1 - orbits[colMap['e']])
        else:
            aval = orbits[colMap['a']]
        period = np.sqrt(aval**3)
        if 'meanAnom' not in colMap:
            meanAnomval = 360.0*(orbits[colMap['epoch']] - orbits[colMap['tPeri']]) / (period*365.25)
        else:
            meanAnomval = orbits[colMap['meanAnom']]
        if 'q' not in colMap:
            qval = orbits[colMap['a']] * (1 - orbits[colMap['e']])
        else:
            qval = orbits[colMap['q']]
        if 'tPeri' not in colMap:
            tPerival = orbits[colMap['epoch']] - (orbits[colMap['meanAnom']]/360.0) * (period*365.25)
        else:
            tPerival = orbits[colMap['tPeri']]

        # Put it all together into a dataframe.
        # Note that parameters which have angular units should be in degrees here.
        self.orbits = pd.DataFrame({'objId':orbids,
                                    'q':qval,
                                    'e':orbits[colMap['e']],
                                    'inc':orbits[colMap['inc']],
                                    'node':orbits[colMap['node']],
                                    'argPeri':orbits[colMap['argPeri']],
                                    'tPeri':tPerival,
                                    'epoch':orbits[colMap['epoch']],
                                    'H':Hval,
                                    'g':gval,
                                    'a':aval,
                                    'M':meanAnomval,
                                    'sed_filename':sedvals})
        self.ssoIds = np.unique(self.orbits['objId'])
        self.nSso = len(self.ssoIds)
