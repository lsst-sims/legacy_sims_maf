# Class for HealpixBinner (healpixel-based spatial binner).
# User can select resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import numpy.ma as ma
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

from .baseSpatialBinner import BaseSpatialBinner
from .baseBinner import BaseBinner


class HealpixBinner(BaseSpatialBinner):
    """Healpix spatial binner."""
    def __init__(self, nside=256, spatialkey1 ='fieldRA' , spatialkey2='fieldDec', verbose=True):
        """Instantiate and set up healpix binner object."""
        super(HealpixBinner, self).__init__(verbose=verbose,
                                            spatialkey1=spatialkey1,spatialkey2=spatialkey2)
        self.badval = hp.UNSEEN 
        # Valid values of nside are powers of 2. 
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise ValueError('Valid values of nside are powers of 2.')
        self.nside = int(nside) 
        self.nbins = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Healpix binner using NSIDE=%d, approximate resolution %f arcminutes' %(self.nside, hp.nside2resol(self.nside, arcmin=True))
        # set variables so binner can be re-constructed
        self.binner_init = {'nside':nside, 'spatialkey1':spatialkey1, 'spatialkey2':spatialkey2}
        self.bins = None 
        
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
        idradec = self.ipix, radec[0], radec[1]
        self.ipix += 1
        return idradec

    def __getitem__(self, ipix):
        """Make healpix binner indexable."""
        idradec = ipix, self._pix2radec(ipix)
        return idradec

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
        lat, lon = hp.pix2ang(self.nside, ipix)
        # Move dec to +/- 90 degrees
        dec = lat - np.pi/2.0
        # Flip ra from latitude to RA (increasing eastward rather than westward)
        ra = -lon % (np.pi*2)
        return ra, dec  

        
    def plotSkyMap(self, metricValue, units=None, title='',
                   clims=None, ylog=False, cbarFormat='%.2g', cmap=cm.jet):
        """Plot the sky map of metricValue using healpy Mollweide plot.

        metricValue = metric values
        units = units for metric color-bar label
        title = title for plot
        
        cbarFormat = format for color bar numerals (i.e. '%.2g', etc) (default to matplotlib default)"""
        # Generate a Mollweide full-sky plot.
        norm = None
        if ylog:
            norm = 'log'
        if cmap is None:
            cmap = cm.jet
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')        
        if clims is not None:
            hp.mollview(metricValue.filled(self.badval), title=title, cbar=False, unit=units, 
                        format=cbarFormat, min=clims[0], max=clims[1], rot=(0,0,180), flip='astro',
                        cmap=cmap, norm=norm)
        else:
            hp.mollview(metricValue.filled(self.badval), title=title, cbar=False, unit=units, 
                        format=cbarFormat, rot=(0,0,180), flip='astro', cmap=cmap, norm=norm)
        hp.graticule(dpar=20., dmer=20.)
        #ecinc = 23.439291 
        #x_ec = np.arange(0, 359., (1.))
        #y_ec = -1*np.sin(x_ec*np.pi/180.) * ecinc
        #hp.projplot(y_ec, x_ec, 'r-', lonlat=True, rot=(180,0,180))
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        ax = plt.gca()
        im = ax.get_images()[0]
        cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                          extend='both', format=cbarFormat)
        cb.set_label(units)
        fig = plt.gcf()
        return fig.number

    def plotHistogram(self, metricValue, title=None, xlabel=None,
                      ylabel='Area (1000s of square degrees)',
                      fignum=None, legendLabel=None, addLegend=False, legendloc='upper left',
                      bins=100, cumulative=False, histRange=None, ylog=False, flipXaxis=False,
                      scale=None):
        """Histogram metricValue over the healpix bin points.

        If scale is None, sets 'scale' by the healpix area per binpoint.
        title = the title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel = y axis label (default 'Area (1000's of square degrees))**
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default 100)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        ylog = use log for y axis (default False)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        # Simply overrides scale and y axis plot label of base plotHistogram. 
        if scale is None:
            scale = (hp.nside2pixarea(self.nside, degrees=True)  / 1000.0)
        fignum = super(HealpixBinner, self).plotHistogram(metricValue, xlabel=xlabel, ylabel=ylabel,
                                                        title=title, fignum=fignum, 
                                                        legendLabel=legendLabel, 
                                                        addLegend=addLegend, legendloc=legendloc,
                                                        bins=bins, cumulative=cumulative,
                                                        histRange=histRange, ylog=ylog,
                                                        flipXaxis=flipXaxis,
                                                        scale=scale)
        return fignum

    def plotPowerSpectrum(self, metricValue, title=None, fignum=None, maxl=500., 
                          legendLabel=None, addLegend=False, removeDipole=True, verbose=False):
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
        cl = hp.anafast(metricValue.filled(self.badval))
        if removeDipole:
            cl = hp.anafast(hp.remove_dipole(metricValue.filled(self.badval), verbose=verbose))
        else:
            cl = hp.anafast(metricValue.filled(self.badval))
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


