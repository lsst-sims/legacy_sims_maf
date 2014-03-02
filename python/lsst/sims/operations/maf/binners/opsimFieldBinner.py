# Class for opsim field based binner.

import numpy as np
import matplotlib.pyplot as plt
import warnings
import lsst.sims.operations.maf.utils as utils

try:
    import astropy.io.fits as pyf
except ImportError:
    import pyfits as pyf
    
from .baseBinner import BaseBinner
from .baseSpatialBinner import BaseSpatialBinner

class OpsimFieldBinner(BaseSpatialBinner):
    """Opsim Field based binner.

    Note that this binner uses the fieldID of the opsim fields to generate spatial matches, thus this
    binner is not generally suitable for use in evaluating dithering or high resolution metrics. """
    def __init__(self, verbose=True):
        super(OpsimFieldBinner, self).__init__(verbose=verbose)
        self.binnertype = 'OPSIMFIELDS'
        self.fieldId = None
        self.ra = None
        self.dec = None
        self.nbins = None

    def setupBinner(self, simData,
                    fieldIdColName='fieldID', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                    useFieldTable=False,
                    dbAddress=None, sessionID=None, fieldTable='Field',
                    proposalTable='tProposal_Field', proposalID=None):
        """Set up opsim field binner object.

        If useFieldTable ==True (default is False), then fetch field information directly from
        Fields (and potentially Fields + Proposal_Field tables) in the opsim (v3) tables, using
        utils.getData.fetchFieldsFromFieldTable.
        
        simData = numpy rec array with simulation pointing history,
        dbAddress = the db connection string
        sessionID = the session ID (numeral section of opsim run name) to use to identify fields, if requesting fields from a particular proposal only
        fieldTable = the name of the db table with the field information (default 'Fields')
        proposalTable = the name of the db table with the proposal information (default 'tProposal_Field')
        proposalID = the proposal ID number (if restricting field choice by fields requested for a proposal) (default = None)
        """
        # Set basic properties for tracking field information, in sorted order.
        if not useFieldTable:
            if self.verbose:
                print 'Using simData to set field information.'
            self.fieldId, idx = np.unique(simData[fieldIdColName], return_index=True)
            self.ra = simData[fieldRaColName][idx]
            self.dec = simData[fieldDecColName][idx]
        else:
            if self.verbose:
                print 'Using Field tables to set field information.'
            fieldData = utils.getData.fetchFieldsFromFieldTable(fieldTable, dbAddress,
                                                                sessionID=sessionID,
                                                                proposalTable=proposalTable,
                                                                proposalID=proposalID)
            idxs = np.argsort(fieldData[fieldIdColName])
            self.fieldId = fieldData[fieldIdColName][idxs]
            self.ra = fieldData[fieldRaColName][idxs]
            self.dec = fieldData[fieldDecColName][idxs]
        self.nbins = len(self.fieldId)
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[fieldIdColName])
        simFieldsSorted = np.sort(simData[fieldIdColName])
        self.left = np.searchsorted(simFieldsSorted, self.fieldId, 'left')
        self.right = np.searchsorted(simFieldsSorted, self.fieldId, 'right')        

    def __iter__(self):
        """Iterate over the binpoints."""
        self.ipix = 0
        return self
    
    def next(self):
        """Return RA/Dec values when iterating over binpoints."""
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.nbins:
            raise StopIteration
        fieldidradec = self.fieldId[self.ipix], self.ra[self.ipix], self.dec[self.ipix]
        self.ipix += 1
        return fieldidradec

    def __getitem__(self, ipix):
        fieldidradec = self.fieldId[ipix], self.ra[self.ipix], self.dec[self.ipix]
        return fieldidradec
    
    def __eq__(self, otherBinner):
        """Evaluate if two grids are equivalent."""
        if isinstance(otherBinner, OpsimFieldBinner):
            return ((np.all(otherBinner.ra == self.ra)) 
                    and (np.all(otherBinner.dec == self.dec))
                    and (np.all(otherBinner.fieldId == self.fieldId)))
        
        else:
            return False

    def sliceSimData(self, binpoint):
        """Slice simData on fieldId, to return relevant indexes for binpoint."""
        i = np.where(binpoint[0] == self.fieldId)
        return self.simIdxs[self.left[i]:self.right[i]]   

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """
        header_dict = dict(comment=comment, metricName=metricName,
                           simDataName=simDataName,
                           metadata=metadata, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval)
        base = BaseBinner()
        base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        #update the header
        hdulist = pyf.open(outfilename, mode='update')
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for key in header_dict.keys():
                    hdulist[0].header[key] = header_dict[key]
        hdulist.close()

        #append the bins
        hdulist = pyf.open(outfilename, mode='append')
        fieldHDU = pyf.PrimaryHDU(data=self.fieldId)
        raHDU = pyf.PrimaryHDU(data=self.ra)
        decHDU =  pyf.PrimaryHDU(data=self.dec)
        hdulist.append(fieldHDU)
        hdulist.append(raHDU)
        hdulist.append(decHDU)
        hdulist.flush()
        hdulist.close()
        return outfilename

    def readMetricData(self, infilename, verbose=False):
        """Read metric values back in and restore the binner"""

        #restore the bins first
        hdulist = pyf.open(infilename)
        if hdulist[0].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        self.fieldId = hdulist[1].data.copy()
        self.ra = hdulist[2].data.copy()
        self.dec = hdulist[3].data.copy() 
        self.nbins = len(self.ra)       
        base = BaseBinner()
        metricValues, header = base.readMetricDataGeneric(infilename)                
        return metricValues, self, header

    # Add some 'rejiggering' to base histogram to make it look nicer for opsim fields.
    def plotHistogram(self, metricValue, title=None, xlabel=None, ylabel='Number of fields',
                      fignum=None, legendLabel=None, addLegend=False, legendloc='upper left',
                      bins=100, cumulative=False, histRange=None, flipXaxis=False,
                      scale=None):
        """Histogram metricValue over the healpix bin points.

        If scale == None, sets 'scale' by the healpix area per binpoint.
        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default 100)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        fignum = super(OpsimFieldBinner, self).plotHistogram(metricValue,  xlabel=xlabel, ylabel=ylabel,
                                                             title=title, fignum=fignum, 
                                                             legendLabel=legendLabel, 
                                                             addLegend=addLegend, legendloc=legendloc,
                                                             bins=bins, cumulative=cumulative,
                                                             histRange=histRange, 
                                                             flipXaxis=flipXaxis, 
                                                             scale=1, yaxisformat='%d')
        return fignum
