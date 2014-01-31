# Class for HealpixBinner (healpixel-based spatial binner).
# User can select resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pyfits as pyf

from .baseSpatialBinner import BaseSpatialBinner
from .baseBinner import BaseBinner

class HealpixBinner(BaseSpatialBinner):
    """Healpix spatial binner."""
    def __init__(self, nside=256, verbose=True):
        """Instantiate and set up healpix binner object."""
        super(HealpixBinner, self).__init__(verbose=verbose)
        self.badval = hp.UNSEEN
        self.binnertype = 'Healpix'
        # Valid values of nside are powers of 2. 
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise Exception('Valid values of nside are powers of 2.')
        self.nside = int(nside) 
        self.nbins = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Healpix binner using NSIDE=%d, approximate resolution %f arcminutes' %(self.nside, hp.nside2resol(self.nside, arcmin=True))
    
    def __iter__(self):
        """Iterate over the binner."""
        self.ipix = 0
        return self

    def next(self):
        """Return RA/Dec values when iterating over bins."""
        # This returns RA/Dec (in radians) of the binpoints. 
        if self.ipix >= self.nbins:
            raise StopIteration
        radec = self._pix2radec(self.ipix)
        self.ipix += 1
        return radec

    def __getitem__(self, ipix):
        """Make healpix binner indexable."""
        radec = self._pix2radec(ipix)
        return radec

    def __eq__(self, otherBinner):
        """Evaluate if two binners are equivalent."""
        # If the two binners are both healpix binners, check nsides value. 
        if isinstance(otherBinner, HealpixBinner):
            return (otherBinner.nside == self.nside)
        else:
            return False
    
    def _pix2radec(self, ipix):
        """Given the pixel number, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix binner.
        # Note that ipix could be an array, 
        # in which case RA/Dec values will be an array also. 
        dec, ra = hp.pix2ang(self.nside, ipix)
        # Move dec to +/- 90 degrees
        dec -= np.pi/2.0
        return ra, dec  

    def writeMetricData(self, outfilename, metricValues,
                        comment='', metricName='',
                        simDataName='', metadata='', 
                        int_badval=-666, badval=-666., dt=np.dtype('float64')):
        """Write metric data and bin data in a fits file """

        header_dict = dict(comment=comment, metricName=metricName, simDataName=simDataName,
                           metadata=metadata, nside=self.nside, binnertype=self.binnertype,
                           dt=dt.name, badval=badval, int_badval=int_badval)
        if metricValues.dtype != 'object': #make a fits file that can be read by ds9
            hp.write_map(outfilename, metricValues)
            hdu=1
            
        else: #if this is a variable length metric, fall back on the generic fits writing
            base = BaseBinner()
            hdu=0
            base.writeMetricDataGeneric(outfilename=outfilename,
                        metricValues=metricValues,
                        comment=comment, metricName=metricName,
                        simDataName=simDataName, metadata=metadata, 
                        int_badval=int_badval, badval=badval, dt=dt)
        #update the header
        hdulist = pyf.open(outfilename, mode='update')
        for key in header_dict.keys():
            hdulist[hdu].header[key] = header_dict[key]
        hdulist.close()
        return outfilename

    def readMetricData(self, infilename):
        """Read metric values back in and restore the binner"""

        hdulist = pyf.open(infilename)
        if 'DT' in hdulist[0].header.keys():
            hdu = 0
        else:
            hdu=1
        dt = hdulist[hdu].header['dt']
        if 'PIXTYPE' in hdulist[hdu].header.keys():
            pixtype = hdulist[hdu].header['PIXTYPE']
        else:
            pixtype=None
        if hdulist[hdu].header['binnertype'] != self.binnertype:
             raise Exception('Binnertypes do not match.')
        hdulist.close()
        if pixtype == 'HEALPIX':
            metricValues, header = hp.read_map(infilename, h=True)
            header = dict(header)
        else:
            base = BaseBinner()
            metricValues, header = base.readMetricDataGeneric(infilename)
        
        #wtf is with the case of header keywords?  Ah, long keywords use HIERATCH cards and preserve case.
        binner = HealpixBinner(nside=header['nside'.upper()])
        binner.badval = header['badval'.upper()]
        binner.int_badval = header['int_badval']
        return metricValues, binner, header
        
    def plotSkyMap(self, metricValue, metricLabel, title='',
                   clims=None, cbarFormat='%.2g'):
        """Plot the sky map of metricValue using healpy Mollweide plot."""
        # Generate a Mollweide full-sky plot.
        if clims!=None:
            hp.mollview(metricValue, title=title, cbar=True, unit=metricLabel, 
                        format=cbarFormat, min=clims[0], max=clims[1], rot=(180,0,180))
        else:
            hp.mollview(metricValue, title=title, cbar=True, unit=metricLabel, 
                        format=cbarFormat, rot=(180,0,180))
        fig = plt.gcf()
        return fig.number

    def plotHistogram(self, metricValue, metricLabel, title=None, 
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
        # Simply overrides scale and y axis plot label of base plotHistogram. 
        if scale == None:
            scale = (hp.nside2pixarea(self.nside, degrees=True)  / 1000.0)
        fignum = super(HealpixBinner, self).plotHistogram(metricValue, metricLabel, 
                                                        title=title, fignum=fignum, 
                                                        legendLabel=legendLabel, 
                                                        addLegend=addLegend, legendloc=legendloc,
                                                        bins=bins, cumulative=cumulative,
                                                        histRange=histRange, 
                                                        flipXaxis=flipXaxis,
                                                        scale=scale)
        plt.ylabel('Area (1000s of square degrees)')
        return fignum

    def plotPowerSpectrum(self, metricValue, title=None, fignum=None, maxl=500., 
                          legendLabel=None, addLegend=False, removeDipole=True):
        """Generate and plot the power spectrum of metricValue.

        maxl = maximum ell value to plot (default 500 .. to plot all l, set to value > 3500)
        title = plot Title (default None)
        fignum = figure number (default None and create new plot)
        legendLabel = label to add in figure legend (default None)
        addLegend = flag to add legend (default False).
        removeDipole = remove dipole when calculating power spectrum (default True) (monopole removed automatically.)
        """
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        cl = hp.anafast(metricValue)
        if removeDipole:
            cl = hp.anafast(hp.remove_dipole(metricValue))
        else:
            cl = hp.anafast(metricValue)
        l = np.arange(np.size(cl))
        # Plot the results.
        if removeDipole:
            condition = ((l < maxl) & (l > 1))
        else:
            condition = (l < maxl)
        plt.plot(l[condition], cl[condition]*l[condition]*(l[condition]+1), label=legendLabel)
        plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l$')
        if addLegend:
            plt.legend(loc='upper right', fancybox=True, prop={'size':'smaller'})
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse/add onto/save this figure if desired). 
        return fig.number


