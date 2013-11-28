# Class for HealpixGrid (healpixel-based spatial grid).
# User can select grid resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from .baseSpatialGrid import BaseSpatialGrid

class HealpixGrid(BaseSpatialGrid):
    """Healpix spatial grid."""
    def __init__(self, nside=256, verbose=True):
        """Set up healpix grid object."""
        # Bad metric data values should be set to badval
        super(HealpixGrid, self).__init__(verbose=verbose)
        self.badval = hp.UNSEEN 
        self._setupGrid(nside = nside)
        return

    def _setupGrid(self, nside=256):
        """Set up healpix grid with nside = nside."""
        # Set up grid. 
        # Valid values of nside are powers of 2. 
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise Exception('Valid values of nside are powers of 2.')
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Set up grid with NSIDE=%d, approximate resolution %f arcminutes' %(self.nside, hp.nside2resol(self.nside, arcmin=True))
        # And we're done for now - don't have to save grid pixel locations or RA/Dec values, as nside describes all necessary information.
        return
    
    def __iter__(self):
        """Iterate over the grid."""
        self.ipix = 0
        return self

    def next(self):
        """Return RA/Dec values when iterating over grid."""
        # To make __iter__ work, you need next. 
        # This returns RA/Dec (in radians) of points in the grid. 
        if self.ipix >= self.npix:
            raise StopIteration
        radec = self.pix2radec(self.ipix)
        self.ipix += 1
        return radec

    def __getitem__(self, ipix):
        """Make healpix grid indexable."""
        radec = self.pix2radec(ipix)
        return radec

    def __eq__(self, otherGrid):
        """Evaluate if two grids are equivalent."""
        # If the two grids are both healpix grids, check nsides value. 
        if isinstance(otherGrid, HealpixGrid):
            return (otherGrid.nside == self.nside)
        else:
            return False
    
    def pix2radec(self, ipix):
        """Given the pixel number, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix grid.
        # Note that ipix could be an array, 
        # in which case RA/Dec values will be an array also. 
        dec, ra = hp.pix2ang(self.nside, ipix)
        # Move dec to +/- 90 degrees
        dec -= np.pi/2.0
        return ra, dec  
    
    def writeFile(self, outfile, metricValues, colHeaders=None):
        """Write metricValues to FITS file using healpy function. """
        # Write metric data values to fits file.
        # If >1 metric values to write in each column, they should be a list of arrays.
        hp.write_map(outfile, metricValues, column_names=colHeaders)
        # If metric values are not an even length of values, this will not 
        #  work and needs some additional handling.
        # TODO : add comments RE sql constraint & metric name
        return
    
    def readFile(self, infile):
        """Read metric data from FITS file using healpy function."""
        # Read metric data values from fits file.
        # First read col0 data and get header to see if there are more columns.
        d, header = hp.read_map(infile, h=True)
        cols = []
        for h in header:
            if h[0].startswith('TTYPE'):
                cols.append(int(h[0].lstrip('TTYPE')) - 1)
            if h[0].startswith('NSIDE'):
                nside = h[1]
        if len(cols) > 1:
            d = hp.read_map(infile, field=cols)
        return nside, d
        
    def plotSkyMap(self, metricValue, metricLabel, title='', 
                   clims=None, cbarFormat='%.2g'):
        """Plot the sky map of metricValue using healpy Mollweide plot."""
        # Generate a Mollweide full-sky plot.
        if clims!=None:
            hp.mollview(metricValue, title=title, cbar=True, unit=metricLabel, 
                        format=cbarFormat, min=clims[0], max=clims[1], rot=(0,0,180))
        else:
            hp.mollview(metricValue, title=title, cbar=True, unit=metricLabel, 
                        format=cbarFormat, rot=(0,0,180))
        fig = plt.gcf()
        return fig.number

    def plotHistogram(self, metricValue, metricLabel, title=None, 
                      fignum=None, legendLabel=None, addLegend=False, 
                      bins=None, cumulative=False, histRange=None, flipXaxis=False,
                      scale=None):
        """Histogram metricValue over the healpix grid points.

        If scale == None, sets 'scale' by the healpix area per gridpoint.
        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, try to set)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        if scale == None:
            scale = (hp.nside2pixarea(self.nside, degrees=True)  / 1000.0)
        fignum = super(HealpixGrid, self).plotHistogram(metricValue, metricLabel, 
                                                        title=title, fignum=fignum, 
                                                        legendLabel=legendLabel, 
                                                        addLegend=addLegend,
                                                        bins=bins, cumulative=cumulative,
                                                        histRange=histRange, 
                                                        flipXaxis=flipXaxis,
                                                        scale=scale)
        return fignum

    def plotPowerSpectrum(self, metricValue, title=None, fignum=None, 
                          label=None, addLegend=False):
        """Generate and plot the power spectrum of metricValue."""
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        # To handle masked values properly, need polespice. (might this work if use_weights & weight values set appropriately?)
        # But this will work when comparing two different angular power spectra calculated in the same way, with the same (incomplete) footprint.
        cl = hp.anafast(metricValue)
        l=np.arange(np.size(cl))
        # Plot the results.
        plt.plot(l,cl*l*(l+1), label=label)
        plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l$')
        if addLegend:
            plt.legend(loc='lower right', fancybox=True, fontsize='smaller')
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse/add onto/save this figure if desired). 
        return fig.number


