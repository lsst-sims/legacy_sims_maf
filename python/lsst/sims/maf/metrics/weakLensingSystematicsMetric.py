import numpy as np
import random
from .metrics import BaseMetric, ExgalM5
from .maps import DustMap
from .slicers import HealpixSlicer
from lsst.sims.utils import angularSeparation
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles


__all__ = ['AverageVisitMetric']

class AverageVisitsMetric(BaseMetric):

    
    def __init__(self,
                 WFDproposalId,
                 runName,
                 maps,
                 Stacker=stackers.RandomDitherFieldPerVisitStacker(
                                        degrees=True),
                 metricName='AverageVisitsMetric',
                 **kwargs):
        """Weak Lensing systematics metric

        Computes the average number of visits per object for 50k objects 
        uniformly distributed within the WFD, after LSS cuts"""
        
        super(AverageVisitsMetric, self).__init__(
            metricName=metricName, col=['fieldId', 'fieldDec'],
            maps=maps, **kwargs
            )
        self.WFDproposalId = WFDproposalId
        self.FOVradius = 1.75
        self.star_num = 5000
        self.counter = {}
        self.positions = []
        self.runName = runName
        self.Stacker = Stacker
        self.stars = self.getPositions(self.runName, self.WFDproposalId)
        self.slice = []

    def run(self, dataSlice, slicePoint=None):
    """runs the metric

    Args:
        dataSlice (namedArrat): positional data from querying the database
        slicePoint (namedArray): queried data along with data from stackers
    Returns:
        None
    """

        for data in dataSlice:
            posRA = data[2]
            posDec = data[3]

            pos = np.array((posRA, posDec))
            self.positions.append(pos)

        return

    def getPositions(self, runName, WFDproposalId):
    """runs ExgalM5 metric to get a random sample of stars
    within the low-extinction, high-depth area.

    Args:
        runName (str): name (and path) of operational simulation.
                       without any extension
        WFDproposalId (int): ID corresponding to the wide-fast-deep
                             main survey in the database

    Returns:
        ndarray: position of stars
    """
        
        opsdb = db.OpsimDatabase(runName+'.db')
        outDir = 'temp'
        resultsDb = db.ResultsDb(outDir=outDir)
        nside = 256
        sqlconstraint = 'filter = "i" and proposalId = ' + WFDproposalId

        metric = ExgalM5(lsstFilter='i')
        dustMap = DustMap(interp=False, nside=nside)
        slicer = HealpixSlicer(nside=nside, useCache=False)

        mBundle = {'temp': metricBundles.MetricBundle(
            metric, 
            slicer, 
            constraint=sqlconstraint,
            stackerList=[], 
            runName=runName,
            metadata='temp to get positions', 
            mapsList=[dustMap])}
        
        bgroup = metricBundles.MetricBundleGroup(mBundle,
                                                 opsdb,
                                                 outDir=outDir,
                                                 resultsDb=resultsDb)
        bgroup.runAll()
        bundle = mBundle['temp']
        cond = np.logical_and.reduce(
            (bundle.slicer.getSlicePoints()['ebv'] < 0.2,
             bundle.metricValues.mask == False,
             bundle.metricValues.data > 26)
        )
        condx = (bundle.slicer.getSlicePoints()['ra'])[cond]
        condy = (bundle.slicer.getSlicePoints()['dec'])[cond]
        centers = [np.array([xi, yi]) for xi, yi in zip(condx, condy)]

        
        return np.array(
            random.sample(centers*100, self.star_num)
            ) + np.random.normal(0, 0.009, (self.star_num, 2))

    def reduceAvgExp(self, _):
    """calculates the number of times each object is observed and returns the mean

    Args:
        None

    Returns:
        float: average number of i-band exposures per object
    """
        self.positions = np.array(self.positions)
        for position_num in range(len(self.stars)):
            star_pos = self.stars[position_num]
            
            cond = angularSeparation(
                self.positions[:, 0],
                self.positions[:, 1],
                star_pos[0]*np.degrees(1),
                star_pos[1]*np.degrees(1)
                ) < self.FOVradius
    
            if tuple(star_pos) in self.counter.keys():
                self.counter[tuple(star_pos)] += len(self.positions[cond])
            else:
                self.counter[tuple(star_pos)] = len(self.positions[cond])
        return np.mean(list(self.counter.values()))
