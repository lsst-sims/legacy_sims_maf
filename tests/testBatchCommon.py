import unittest
import lsst.utils.tests
import lsst.sims.maf.batches as batches


class TestCommon(unittest.TestCase):

    def testFilterList(self):
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=False, extraSql=None)
        self.assertEqual(len(filterlist), 6)
        self.assertEqual(len(colors), 6)
        self.assertEqual(sqls['u'], 'filter = "u"')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql=None)
        self.assertIn('all', filterlist)
        self.assertEqual(sqls['all'], '')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql='night=3')
        self.assertEqual(sqls['all'], 'night=3')
        self.assertEqual(sqls['u'], '(night=3) and (filter = "u")')
        self.assertEqual(metadata['u'], 'night=3 u band')
        filterlist, colors, orders, sqls, metadata = batches.common.filterList(all=True, extraSql='night=3',
                                                                               extraMetadata='night 3')
        self.assertEqual(metadata['u'], 'night 3 u band')

class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
