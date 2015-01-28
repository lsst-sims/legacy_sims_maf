import matplotlib
matplotlib.use("Agg")
import unittest
import numpy as np

import lsst.sims.maf.utils as utils

class TestUtils(unittest.TestCase):

    def testStellarMags(self):
        """
        Test that the stellar magnitudes are created for accepted types.
        """
        stellarTypes = ['O','B','A','F','G','K','M']
        for st in stellarTypes:
            mags = utils.stellarMags(st)
            mags2 = utils.stellarMags(st, rmag=20.)
            for key in mags:
                self.assertLess(mags[key], mags2[key])
        self.assertRaises(ValueError, utils.stellarMags, 'ack')

    def testCreateSqlWhere(self):
        """
        Test that the createSQLWhere method handles expected cases.
        """
        # propTags is a dictionary of lists returned by OpsimDatabase
        propTags = {'WFD':[1, 2, 3], 'DD':[4], 'Rolling':[2]}
        # If tag is in dictionary with one value, returned sql where clause
        #  is simply 'propId = 4'
        tag = 'DD'
        sqlWhere = utils.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, 'propID = 4')
        # if multiple proposals with the same tag, all should be in list.
        tag = 'WFD'
        sqlWhere = utils.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere.split()[0], '(propID')
        for id in propTags['WFD']:
            self.assertTrue('%s' %(id) in sqlWhere)
        # And the same id can be in multiple proposals.
        tag = 'Rolling'
        sqlWhere = utils.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, 'propID = 2')
        # And tags not in propTags are handled.
        badprop = 'propID like "NO PROP"'
        tag = 'nogo'
        sqlWhere = utils.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, badprop)
        # And tags which identify no proposal ID are handled.
        propTags['Rolling'] = []
        sqlWhere = utils.createSQLWhere(tag, propTags)
        self.assertEqual(sqlWhere, badprop)


if __name__ == "__main__":
    unittest.main()
