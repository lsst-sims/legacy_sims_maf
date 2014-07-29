import os
import warnings
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
from lsst.sims.maf.db import ResultsDb


class BaseSliceMetric(object):
    """
    The BaseSlicemetric class provides base functionality common to all
    sliceMetrics. 
    A 'sliceMetric' in general couples slicers and metrics, and provides
    storage for things like metric data and metadata about the metric + slicer.
    """
    def __init__(self, useResultsDb=True, resultsDbAddress=None, 
                 figformat='pdf', thumbnail=True, dpi=600, outDir='Output'):
        """
        Instantiate sliceMetric object and set up (empty) dictionaries.
        The dictionaries are keyed by an internal-use id number. """
        # Track output directory.
        self.outDir = outDir
        self.thumbnail = thumbnail
        # Set up results database storage if desired. 
        if useResultsDb:
           self.resultsDb = ResultsDb(outDir=self.outDir, 
                                      resultsDbAddress=resultsDbAddress)
           # If we're using the resultsDb, track the metricID's used there.
           self.metricIds = {} 
        else:
           self.resultsDb = False
        # Set figure format for output plot files.
        self.figformat = figformat
        self.dpi = dpi
        # Set up dictionaries to store metric data, slicer info, and 
        #  metadata. Keyed by a unique internal id# (iid). 
        # Note that metricNames are not necessarily unique by themselves.
        self.iid_next = 0
        self.metricNames = {}
        self.plotParams = {}
        self.displayDicts = {}        
        self.slicers = {}
        self.metricValues = {}
        self.simDataNames = {}
        self.sqlconstraints = {}
        self.metadatas = {}


    def metricNameIid(self, metricName):
        """
        Return the internal dictionary id number  (iid) for a given metricName.
        
        If metricName is a duplicate, will return all iids which match.
        """
        iids = []
        for iid, name in self.metricNames.iteritems():
            if name == metricName:
                iids.append(iid)
        return iids

    def findIids(self, simDataName=None, metricName=None, metadata=None, slicerName=None):
        """
        Identify iids which match simDataName/metricName/metadata/slicer.
        """
        iids = self.metricValues.keys()
        for iid in self.metricValues.keys():
           if iid in iids:
              if simDataName is not None:
                 if self.simDataNames[iid] != simDataName:
                    iids.remove[iid]
                    continue
              if metricName is not None:
                 if self.metricNames[iid] != metricName:
                    iids.remove[iid]
                    continue
              if metadata is not None:
                 if self.metadatas[iid] != metadata:
                    iids.remove[iid]
                    continue
              if slicerName is not None:
                 if self.slicers[iid].slicerName != slicerName:
                    iids.remove[iid]
                    continue
        return iids
    
       
    def _buildOutfileName(self, iid, outfileRoot=None, plotType=None):
        """
        Build an automatic output file name for metric data or plots.
        """
        # Start building output file name, using either root provided or runName.
        oname = outfileRoot
        if oname is None:
            if iid in self.simDataNames:
                oname = self.simDataNames[iid]
            else:
                oname = list(set(self.simDataNames.values()))
                if len(oname) > 1: # more than one runName -- punt.
                    oname = 'comparison'
                else: # all the runNames matched, so let's just use that.
                    oname = oname[0]
        # Add metric name.
        if iid in self.metricNames:
            oname = oname + '_' + self.metricNames[iid]
        # Add summary of the metadata if it exists.
        if iid in self.metadatas:
            oname = oname + '_' + self.metadatas[iid]
        # Add letters to distinguish slicer types
        if iid in self.slicers:
            oname = oname + '_' + self.slicers[iid].slicerName[:4].upper()
        # Do some work sanitizing output filename. 
        # Replace <, > and = signs.
        oname = oname.replace('>', 'gt').replace('<', 'lt').replace('=', 'eq')
        # Strip white spaces (replace with underscores), strip '.'s and ','s
        oname = oname.replace('  ', ' ').replace(' ', '_').replace('.', '_').replace(',', '')
        # and strip quotes and double __'s
        oname = oname.replace('"','').replace("'",'').replace('__', '_')
        # and remove / and \
        oname = oname.replace('/', '_').replace('\\', '_')
        # and remove parentheses
        oname = oname.replace('(', '').replace(')', '')
        # Add plot name, if plot.
        if plotType is not None:
           oname = oname + '_' + plotType + '.' + self.figformat
        return oname

    
    def readMetricData(self, filenames, verbose=False):
       """
       Given a list of filenames, reads metric values and metadata from disk. 
       """
       if not hasattr(filenames, '__iter__'):
          filenames = [filenames, ]        
       for f in filenames:
          # Set up a base slicer to read data.
          baseslicer = slicers.BaseSlicer()
          metricData, slicer, header = baseslicer.readData(f)
          iid = self.iid_next
          self.iid_next += 1
          self.slicers[iid] = slicer
          self.metricValues[iid] = metricData
          self.metricValues[iid].fill_value = slicer.badval
          self.metricNames[iid] = header['metricName']
          self.simDataNames[iid] = header['simDataName']
          self.sqlconstraints[iid] = header['sqlconstraint']
          self.metadatas[iid] = header['metadata']
          self.plotParams[iid] = {}
          # Set default values, in  case metric file doesn't have the info.
          self.displayDicts[iid] = {'group':'Ungrouped', 
                                    'subgroup':'NULL',
                                    'order':0,
                                    'caption':'NULL'}
          if 'displayDict' in header:
              self.displayDicts[iid].update(header['displayDict'])
          if 'plotParams' in header:
             self.plotParams[iid].update(header['plotParams'])
          if verbose:
             print 'Read data from %s, got metric data for metricName %s' %(f, header['metricName'])
            
    def writeAll(self, outfileRoot=None, comment=''):
       """
       Write all metric values to disk.
       """
       for iid in self.metricValues:
          outfilename = self.writeMetric(iid, comment=comment,
                                         outfileRoot=outfileRoot)
        
    def writeMetric(self, iid, comment='', outfileRoot=None):
        """
        Write self.metricValues[iid] (and associated metadata) to disk.

        comment = any additional comments to add to output file (beyond 
                   metric name, simDataName, and metadata).
        outfileRoot = root of the output files (default simDataName).
       """
        outfile = self._buildOutfileName(iid, outfileRoot=outfileRoot)
        outfile = outfile + '.npz'
        if iid in self.slicers:
           slicer = self.slicers[iid]
        else:
           try:
              slicer = self.slicer
           except AttributeError:
              # Otherwise, try just saving with base slicer.
              # This won't save any metadata about what the slices looked like.
              slicer = slicers.BaseSlicer()
        slicer.writeData(os.path.join(self.outDir, outfile),
                         self.metricValues[iid],
                         metricName = self.metricNames[iid],
                         simDataName = self.simDataNames[iid],
                         sqlconstraint = self.sqlconstraints[iid],
                         metadata = self.metadatas[iid] + comment,
                         displayDict = self.displayDicts[iid])
        if self.resultsDb:
            self.metricIds[iid] = self.resultsDb.addMetric(self.metricNames[iid],
                                                          slicer.slicerName,
                                                          self.simDataNames[iid],
                                                          self.sqlconstraints[iid],
                                                          self.metadatas[iid],
                                                          outfile,
                                                          self.displayDicts[iid])
           
