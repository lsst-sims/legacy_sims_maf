import os, warnings
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from .baseSliceMetric import BaseSliceMetric

from lsst.sims.maf.utils import ColInfo

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())

class RunSliceMetric(BaseSliceMetric):
    """
    RunSliceMetric couples a single slicer and multiple metrics, in order
    to generate metric data values at all points over the slicer.

    Given the list of metrics and slicers (plus customized stackers, if desired),
    and the sql constraint relevant for this slicer/metric calculation,
    the RunSliceMetric will discover the list of columns required and get this information
    from the database.

    The RunSliceMetric handles metadata about each metric, including the
    opsim run name, the sql constraint on the query used to obtain the input data,
    and the slicer type that produced the metric data.
    """
    def __init__(self, useResultsDb=True, resultsDbAddress=None,
                 figformat='pdf', dpi=600, outDir='Output', thumbnail=True):
        """
        Instantiate the RunSliceMetric.
        """
        super(RunSliceMetric, self).__init__(useResultsDb=useResultsDb, resultsDbAddress=resultsDbAddress,
                                             figformat=figformat, dpi=dpi, outDir=outDir, thumbnail=True)
        # Add dictionary to save metric objects
        self.metricObjs = {}
        self.slicer = None
        self.stackerObjs = set()
        self.thumbnail = thumbnail

    def getMetricObjIid(self, metricObj):
       """
       Return the internal dictionary id number (iid) for a given metricObject.

       If metricObj is a duplicate, will return all iids which match.
       """
       iids = []
       for iid, metric in self.metricObjs.iteritems():
          if metric == metricObj:
             iids.append(iid)
       return iids

    def setMetricsSlicerStackers(self, metricList, slicer, stackerList=None):
        """
        Set the metric objects used to calculate metric values, together with the
        slicer that will be used here, and optionally set the customized stackers.
        Note that each of these objects must be instantiated by the user.
        """
        self._setSlicer(slicer, override=True)
        self._setMetrics(metricList)
        self._setStackers(stackerList)

    def _setSlicer(self, slicer, override=False):
        """
        Set slicer for RunSliceMetric.

        If override = False (default), checks if slicer already set, and if the two are equal.
        """
        if (self.slicer is None) or (override):
            self.slicer = slicer
            # Update existing slicer dict
            for iid in self.slicers:
                self.slicers[iid] = self.slicer
            return True
        return (self.slicer == slicer)

    def _setMetrics(self, metricList):
        """
        Set the metric objects used to calculate metric values.
        Need to set slicer first!
        """
        if not hasattr(metricList, '__iter__'):
            metricList = [metricList,]
        iid = self.iid_next
        for metric in metricList:
            self.metricObjs[iid] = metric
            self.plotDicts[iid] = metric.plotDict
            self.displayDicts[iid] = metric.displayDict
            self.metricNames[iid] = metric.name
            self.slicers[iid] = self.slicer
            iid += 1
        self.iid_next = iid

    def _setStackers(self, stackerList=None):
        """
        Set the stackers.
        Note that these stackers are not tied to particular metrics -- they are just
         used to generate extra data from the database.
        """
        if not hasattr(stackerList, '__iter__'):
            stackerList = [stackerList,]
        for s in stackerList:
            if s is not None:
                self.stackerObjs.add(s)

    def findDataCols(self):
        """
        Determine the complete list of columns that must be queried from the database.

        Uses the metrics, slicer, and stackers to determine the necessary columns, returns this list.
        Instantiates any additional necessary stackers.
        """
        # Find the columns required  by the metrics and slicers (including if they come from stackers).
        colInfo = ColInfo()
        dbcolnames = set()
        defaultstackers = set()
        # Look for the source for the columns for the slicer.
        for col in self.slicer.columnsNeeded:
            colsource = colInfo.getDataSource(col)
            if colsource != colInfo.defaultDataSource:
                defaultstackers.add(colsource)
            else:
                dbcolnames.add(col)
        # Look for the source of columns in the metrics.
        for col in self.metricObjs[0].colRegistry.colSet:
            colsource = colInfo.getDataSource(col)
            if colsource != colInfo.defaultDataSource:
                defaultstackers.add(colsource)
            else:
                dbcolnames.add(col)
        # Remove explicity instantiated stackers from defaultstacker set.
        for s in self.stackerObjs:
            if s.__class__ in defaultstackers:
                defaultstackers.remove(s.__class__)
        # Instantiate and add the remaining default stackers.
        for s in defaultstackers:
            self.stackerObjs.add(s())
        # Add the columns needed by all stackers to the list to grab from the database.
        for s in self.stackerObjs:
            for col in s.colsReq:
                dbcolnames.add(col)
        return dbcolnames


    def validateMetricData(self, simData):
        """
        Validate that simData has the required data values for the metrics in self.metricObjs.
        """
        simCols = self.metricObjs[0].colRegistry.colSet
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n')
        return True

    def runSlices(self, simData, simDataName='opsim', sqlconstraint='', metadata='',
                  fieldData=None, maps=None):
        """
        Generate metric values, iterating through self.slicer and running self.metricObjs for each slice.

        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data (i.e. the opsim run name).
        sqlconstraint = the sql where clause used to pull data from simDataName.
        metadata = further information from config files ('WFD', 'r band', etc.).
        fieldData = numpy recarray holding the information on the field pointings -- used for OpsimFieldSlicer ONLY
        maps = skymap (such as dust extinction map) objects to add to slicer metadata at each slicepoint
        """
        # Set up indexing in slicer.
        if self.slicer.slicerName == 'OpsimFieldSlicer':
            if fieldData is None:
                raise ValueError('For opsimFieldSlicer, need to provide fieldData to setup slicer')
            self.slicer.setupSlicer(simData, fieldData, maps=maps)
        else:
            self.slicer.setupSlicer(simData, maps=maps)
        # Set simDataName, sqlconstraint and metadata for each metric.
        for iid in self.metricObjs:
           self.simDataNames[iid] = simDataName
           self.sqlconstraints[iid] = sqlconstraint
           self.metadatas[iid] = metadata
           if len(self.metadatas[iid]) == 0:
              self.metadatas[iid] = self.sqlconstraints[iid]
        # Set up (masked) arrays to store metric data.
        for iid in self.metricObjs:
           self.metricValues[iid] = ma.MaskedArray(data = np.empty(len(self.slicer),
                                                                   self.metricObjs[iid].metricDtype),
                                                   mask = np.zeros(len(self.slicer), 'bool'),
                                                   fill_value=self.slicer.badval)
        # Set up an ordered dictionary to be the cache if needed:
        # (Currently using OrderedDict, it might be faster to use 2 regular Dicts instead)
        if self.slicer.cacheSize > 0:
           cacheDict = OrderedDict()
           cache = True
        else:
           cache = False
        # Run through all slicepoints and calculate metrics.
        for i, slice_i in enumerate(self.slicer):
            slicedata = simData[slice_i['idxs']]
            if len(slicedata)==0:
                # No data at this slicepoint. Mask data values.
               for iid in self.metricObjs:
                  self.metricValues[iid].mask[i] = True
            else:
               # There is data! Should we use our data cache?
               if cache:
                  # Make the data idxs hashable.
                  key = str(sorted(slice_i['idxs']))[1:-1].replace(' ','')
                  # If key exists, set flag to use it, otherwise add it
                  if key in cacheDict:
                     useCache = True
                  else:
                     cacheDict[key] = i
                     useCache = False
                     # If we are above the cache size, drop the oldest element from the cache dict
                     if i > self.slicer.cacheSize:
                        cacheDict.popitem(last=False) #remove 1st item
                  for iid in self.metricObjs:
                     if useCache:
                        self.metricValues[iid].data[i] = self.metricValues[iid].data[cacheDict[key]]
                     else:
                        self.metricValues[iid].data[i] = self.metricObjs[iid].run(slicedata,
                                                                                  slicePoint=slice_i['slicePoint'])
               # Not using memoize, just calculate things normally
               else:
                  for iid in self.metricObjs:
                     self.metricValues[iid].data[i] = self.metricObjs[iid].run(slicedata,
                                                                               slicePoint=slice_i['slicePoint'])

        # Mask data where metrics could not be computed (according to metric bad value).
        for iid in self.metricObjs:
           if self.metricValues[iid].dtype.name == 'object':
              for ind,val in enumerate(self.metricValues[iid].data):
                 if val is self.metricObjs[iid].badval:
                    self.metricValues[iid].mask[ind] = True
           else:
              # For some reason, this doesn't work for dtype=object arrays.
              self.metricValues[iid].mask = np.where(self.metricValues[iid].data==self.metricObjs[iid].badval,
                                                     True, self.metricValues[iid].mask)


    def reduceAll(self):
        """
        Run all reduce functions on all (complex) metrics.
        """
        for iid in self.metricObjs:
            # If there are no reduce functions, skip this metric.
            if len(self.metricObjs[iid].reduceFuncs.keys()) ==0:
                continue
            # Apply reduce functions.
            self.reduceMetric(iid, self.metricObjs[iid].reduceFuncs.values(),
                              self.metricObjs[iid].reduceOrder.values())

    def reduceMetric(self, iid, reduceFunc, reduceOrder=None):
        """
        Run 'reduceFunc' (method on metric object) on self.metricValues[iid].

        reduceFunc can be a list of functions to be applied to the same metric data.
        reduceOrder can be list of integers to add to the displayDict['order'] value for each
          reduced metric value (can also be None).
        """
        if not isinstance(reduceFunc, list):
            reduceFunc = [reduceFunc,]
        # Autogenerate metric reduced value names.
        rNames = []
        metricName = self.metricNames[iid]
        for r in reduceFunc:
            rNames.append(metricName + '_' + r.__name__.replace('reduce',''))
        # Make sure reduceOrder is available.
        if reduceOrder is None:
            reduceOrder = np.zeros(len(reduceFunc), int)
        if len(reduceOrder) < len(reduceFunc):
            rOrder = np.zeros(len(reduceFunc), int) + len(reduceFunc)
            for i, r in enumerate(reduceOrder):
                rOrder[i] = r
            reduceOrder = rOrder.copy()
        # Set up reduced metric values masked arrays, copying metricName's mask,
        # and copy metadata/plotparameters, etc.
        riids = np.arange(self.iid_next, self.iid_next+len(rNames), 1)
        self.iid_next = riids.max() + 1
        for riid, rName, rOrder in zip(riids, rNames, reduceOrder):
           self.metricNames[riid] = rName
           self.slicers[riid] = self.slicer
           self.simDataNames[riid] = self.simDataNames[iid]
           self.sqlconstraints[riid] = self.sqlconstraints[iid]
           self.metadatas[riid] = self.metadatas[iid]
           self.plotDicts[riid] = self.plotDicts[iid]
           self.displayDicts[riid] = self.displayDicts[iid].copy()
           self.displayDicts[riid]['order'] = self.displayDicts[riid]['order'] + rOrder
           self.metricValues[riid] = ma.MaskedArray(data = np.empty(len(self.slicer), 'float'),
                                                    mask = self.metricValues[iid].mask,
                                                    fill_value=self.slicer.badval)
        # Run through slicepoints/metricValues applying all reduce functions.
        for i, (mVal, mMask) in enumerate(zip(self.metricValues[iid].data, self.metricValues[iid].mask)):
           if not mMask:
              # Evaluate reduced version of metric values.
              for riid, rFunc in zip(riids, reduceFunc):
                 self.metricValues[riid].data[i] = rFunc(mVal)



    def writeMetric(self, iid, comment='', outfileRoot=None, outfileSuffix=None):
        """
        Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
       """
        super(RunSliceMetric, self).writeMetric(iid, comment=comment, outfileRoot=outfileRoot,
                                                outfileSuffix=outfileSuffix)
        # Have to return additional info for driver merged histograms .. need to update this later.
        if iid in self.metricObjs:
            outfile = self._buildOutfileName(iid, outfileRoot=outfileRoot, outfileSuffix=outfileSuffix) + '.npz'
            self.metricObjs[iid].saveFile = outfile
