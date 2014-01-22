# Class for HealpixBinner (healpixel-based spatial binner).
# User can select resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from .baseSpatialBinner import BaseSpatialBinner

class HealpixBinner(BaseSpatialBinner):
    """Healpix spatial binner."""
    def __init__(self, nside=256, verbose=True):
        """Instantiate and set up healpix binner object."""
        super(HealpixBinner, self).__init__(verbose=verbose)
        self.badval = hp.UNSEEN 
        # Valid values of nside are powers of 2. 
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise Exception('Valid values of nside are powers of 2.')
        self.nside = nside
        self.nbins = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Set up binner with NSIDE=%d, approximate resolution %f arcminutes' %(self.nside, hp.nside2resol(self.nside, arcmin=True))
    
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
                          legendLabel=None, addLegend=False):
        """Generate and plot the power spectrum of metricValue.

        maxl = maximum ell value to plot (default 500 .. to plot all l, set to value > 3500)
        title = plot Title (default None)
        fignum = figure number (default None and create new plot)
        legendLabel = label to add in figure legend (default None)
        addLegend = flag to add legend (default False).
        """
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
        # To handle masked values properly, need polespice. (might this work if use_weights & weight values set appropriately?)
        # But this will work when comparing two different angular power spectra calculated in the same way, with the same (incomplete) footprint.
        cl = hp.anafast(metricValue)
        l=np.arange(np.size(cl))
        # Plot the results.
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


