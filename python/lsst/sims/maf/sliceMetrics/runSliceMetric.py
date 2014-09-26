import os
import warnings
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.db import ResultsDb
from .baseSliceMetric import BaseSliceMetric

import time
def dtime(time_prev):
   return (time.time() - time_prev, time.time())


class RunSliceMetric(BaseSliceMetric):
    """
    RunSliceMetric couples a single slicer and multiple metrics, in order
    to generate metric data values at all points over the slicer.
    
    The RunSliceMetric handles metadata about each metric, including the
    opsim run name, the sql constraint on the query used to obtain the input data,
    and the slicer type that produced the metric data.
    """
    def __init__(self, useResultsDb=True, resultsDbAddress=None,
                 figformat='pdf', dpi=600, outDir='Output'):
        """
        Instantiate the RunSliceMetric.
        """
        super(RunSliceMetric, self).__init__(useResultsDb=useResultsDb, resultsDbAddress=resultsDbAddress,
                                             figformat=figformat, dpi=dpi, outDir=outDir)
        # Add dictionary to save metric objects
        self.metricObjs = {}
        self.slicer = None

    def metricObjIid(self, metricObj):
       """
       Return the internal dictionary id number (iid) for a given metricObject.

       If metricObj is a duplicate, will return all iids which match.
       """
       iids = []
       for iid, metric in self.metricObjs.iteritems():
          if metric == metricObj:
             iids.append(iid)
       return iids
    
    def setSlicer(self, slicer, override=False):
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

    def setMetrics(self, metricList):
        """
        Set the metric objects used to calculate metric values.
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
        return

    def validateMetricData(self, simData):
        """
        Validate that simData has the required data values for the metrics in self.metricObjs.
        """
        simCols = self.metricObjs[0].colRegistry.colSet
        for c in simCols:
            if c not in simData.dtype.names:
                raise Exception('Column', c,'not in simData: needed by the metrics.\n')
        return True

    def runSlices(self, simData, simDataName='opsim', sqlconstraint='', metadata=''):
        """
        Generate metric values, iterating through self.slicer and running self.metricObjs for each slice.

        simData = numpy recarray holding simulated data
        simDataName = identifier for simulated data (i.e. the opsim run name).
        sqlconstraint = the sql where clause used to pull data from simDataName.
        metadata = further information from config files ('WFD', 'r band', etc.).
        """
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
                        pop = cacheDict.popitem(last=False) #remove 1st item
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
            # Apply reduce functions 
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


    def computeSummaryStatistics(self, iid, summaryMetric):
        """
        Compute single number summary of self.metricValues[iid], using summaryMetric.

        summaryMetric must be an object (not a class), already instantiated.
        """
        if not hasattr(iid, '__iter__'):
            iid = [iid,]
        summaryValues = []
        for iidi in iid:
            # To get (clear, non-confusing) result from unislicer, try running this with 'Identity' metric.
            # Create numpy structured array from metric data, with bad values removed.
            rarr = np.array(zip(self.metricValues[iidi].compressed()),
                            dtype=[('metricdata', self.metricValues[iidi].dtype)])
            # The summary metric colname should already be set to 'metricdata', but in case it's not:
            summaryMetric.colname = 'metricdata'
            if np.size(rarr) == 0:
               summaryValue = self.slicer.badval
            else:
               summaryValue = summaryMetric.run(rarr)
            summaryValues.append(summaryValue)
            # Add summary metric info to results database. (should be float or int).
            if self.resultsDb:
                if iidi not in self.metricIds:
                    self.metricIds[iidi] = self.resultsDb.addMetric(self.metricNames[iidi], self.slicer.slicerName,
                                                                    self.simDataNames[iidi],
                                                                    self.sqlconstraints[iidi],
                                                                    self.metadatas[iidi], None)
                self.resultsDb.addSummaryStat(self.metricIds[iidi],
                                                summaryName=summaryMetric.name.replace(' metricdata', ''),
                                                summaryValue=summaryValue)
        return summaryValues


    def writeMetric(self, iid, comment='', outfileRoot=None):
        """
        Write metric values 'metricName' to disk.

        comment = any additional comments to add to output file (beyond
           metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
       """
        super(RunSliceMetric, self).writeMetric(iid, comment=comment, outfileRoot=outfileRoot)
        # For driver merged histograms .. need to update this later.
        if iid in self.metricObjs:
            outfile = self._buildOutfileName(iid, outfileRoot=outfileRoot) + '.npz'
            self.metricObjs[iid].saveFile = outfile


    def captionMetric(self, iid):
        """
        Auto generate caption for a given metric.
        """
        displayOrder = ['plotSkyMap', 'plotHistogram', 'plotPowerSpectrum']
        if (self.displayDicts[iid]['caption'] is None) or \
            (self.displayDicts[iid]['caption'].endswith('(auto)')):
          caption = ''
          plotTypes = self.slicer.plotFuncs.keys()
          if len(plotTypes) > 0:
            caption += 'Plots ('
            ptypes = []
            for p in displayOrder:
                if p in plotTypes:
                    ptypes.append(p)
                    plotTypes.remove(p)
            for r in plotTypes:
                ptypes.append(r)
            for p in ptypes:
                caption += '%s, ' %(p.replace('plot', ''))
            caption = caption[:-2] + ') for '
          caption += '%s ' %(self.metricNames[iid])
          caption += 'calculated with a %s slicer ' %(self.slicer.slicerName)
          if len(self.metadatas[iid].strip()) > 0:
            caption += 'on a subset of data selected in %s. ' %(self.metadatas[iid].strip())
          if 'zp' in self.plotDicts[iid]:
            caption += 'Values plotted with a zeropoint of %.2f. ' %(self.plotDicts[iid]['zp'])
          if 'normVal' in self.plotDicts[iid]:
            caption += 'Values plotted with a normalization value of %.2f. ' %(self.plotDicts[iid]['normVal'])
          caption += '(auto)'
          self.displayDicts[iid]['caption'] = caption
        if self.resultsDb:
          if iid not in self.metricIds:
            self.metricIds[iid] = self.resultsDb.addMetric(self.metricNames[iid], self.slicer.slicerName,
                                                            self.simDataNames[iid], self.sqlconstraints[iid],
                                                            self.metadatas[iid], None)
          if self.displayDicts[iid]['subgroup'] is None:
             self.displayDicts[iid]['subgroup'] = self.slicer.slicerName
          self.resultsDb.addDisplay(self.metricIds[iid], self.displayDicts[iid])

    def plotAll(self, savefig=True, closefig=False, outfileRoot=None, verbose=True):
        """
        Plot histograms and skymaps (where relevant) for all metrics.
        """
        for iid in self.metricValues:
           plotfigs = self.plotMetric(iid, savefig=savefig, outfileRoot=outfileRoot)
           if closefig:
              plt.close('all')
           if plotfigs is None and verbose:
              warnings.warn('Not plotting metric data for %s' %(mname))

    def plotMetric(self, iid, savefig=True, outfileRoot=None):
        """
        Create all plots for 'metricName' .
        """
        # Get the metric plot parameters.
        pParams = self.plotDicts[iid].copy()
        # Build plot title and label.
        mname = self.metricNames[iid]
        # "Units" always in pParams, but might be '' (== the physical units).
        if 'title' not in pParams:
            # Build default title.
            pParams['title'] = self.simDataNames[iid] + ' ' + self.metadatas[iid]
            pParams['title'] += ': ' + mname
        if 'ylabel' not in pParams:
            # Build default y label if needed (i.e. oneDSlicer)
            if self.slicer.slicerName == 'OneDSlicer':
                pParams['ylabel'] = mname + ' (' + pParams['units'] + ')'
        if 'xlabel' not in pParams:
            # Build a default x label if needed
            if self.slicer.slicerName == 'OneDSlicer':
                pParams['xlabel'] = self.slicer.sliceColName + ' (' + self.slicer.sliceColUnits + ')'
            else:
                pParams['xlabel'] = mname + ' (' + pParams['units'] + ')'
        # Plot the data.
        # Plotdata for each slicer returns a dictionary with the filenames, filetypes, and fig nums.
        outfile = self._buildOutfileName(iid, outfileRoot=outfileRoot)
        plotResults = self.slicer.plotData(self.metricValues[iid], savefig=savefig,
                                           figformat=self.figformat, dpi=self.dpi,
                                           filename=os.path.join(self.outDir, outfile), **pParams)
        # Save information about the plotted files.
        if self.resultsDb:
            if iid not in self.metricIds:
                self.metricIds[iid] = self.resultsDb.addMetric(self.metricNames[iid], self.slicer.slicerName,
                                                               self.simDataNames[iid], self.sqlconstraints[iid],
                                                               self.metadatas[iid], None)
            self.captionMetric(iid)
            for filename, filetype in zip(plotResults['filenames'], plotResults['filetypes']):
                froot, fname = os.path.split(filename)
                self.resultsDb.addPlot(metricId=self.metricIds[iid], plotType=filetype, plotFile=fname)
        return plotResults['figs']
