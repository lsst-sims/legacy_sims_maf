# Class for HealpixSlicer (healpixel-based spatial slicer).
# User can select resolution using 'NSIDE'
# Requires healpy
# See more documentation on healpy here http://healpy.readthedocs.org/en/latest/tutorial.html
# Also requires numpy and pylab (for histogram and power spectrum plotting)

import numpy as np
import numpy.ma as ma
import warnings
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from lsst.sims.maf.utils import percentileClipping

from .baseSpatialSlicer import BaseSpatialSlicer
from .baseSlicer import BaseSlicer


class HealpixSlicer(BaseSpatialSlicer):
    """Healpix spatial slicer."""
    def __init__(self, nside=128, spatialkey1 ='fieldRA' , spatialkey2='fieldDec', verbose=True, 
                 useCache=True, radius=1.75, leafsize=100):
        """Instantiate and set up healpix slicer object."""
        super(HealpixSlicer, self).__init__(verbose=verbose,
                                            spatialkey1=spatialkey1, spatialkey2=spatialkey2,
                                            badval=hp.UNSEEN, radius=radius, leafsize=leafsize) 
        # Valid values of nside are powers of 2. 
        # nside=64 gives about 1 deg resolution
        # nside=256 gives about 13' resolution (~1 CCD)
        # nside=1024 gives about 3' resolution
        # Check validity of nside:
        if not(hp.isnsideok(nside)):
            raise ValueError('Valid values of nside are powers of 2.')
        self.nside = int(nside)
        self.pixArea = hp.nside2pixarea(self.nside)
        self.nslice = hp.nside2npix(self.nside)
        if self.verbose:
            print 'Healpix slicer using NSIDE=%d, approximate resolution %f arcminutes' %(self.nside,
                                                                                          hp.nside2resol(self.nside,
                                                                                                         arcmin=True))
        # Set variables so slicer can be re-constructed
        self.slicer_init = {'nside':nside, 'spatialkey1':spatialkey1, 'spatialkey2':spatialkey2,
                            'radius':radius}
        if useCache:
            # useCache set the size of the cache for the memoize function in sliceMetric.
            binRes = hp.nside2resol(nside) # Pixel size in radians
            # Set the cache size to be ~2x the circumference
            self.cacheSize = int(np.round(4.*np.pi/binRes))
        # Set up slicePoint metadata.
        self.slicePoints['sid'] = np.arange(self.nslice)
        self.slicePoints['ra'], self.slicePoints['dec'] = self._pix2radec(self.slicePoints['sid'])        

    def __eq__(self, otherSlicer):
        """Evaluate if two slicers are equivalent."""
        # If the two slicers are both healpix slicers, check nsides value. 
        if isinstance(otherSlicer, HealpixSlicer):
            return (otherSlicer.nside == self.nside)
        else:
            return False

    def _pix2radec(self, islice):
        """Given the pixel number / sliceID, return the RA/Dec of the pointing, in radians."""
        # Calculate RA/Dec in RADIANS of pixel in this healpix slicer.
        # Note that ipix could be an array, 
        # in which case RA/Dec values will be an array also. 
        lat, lon = hp.pix2ang(self.nside, islice)
        # Move dec to +/- 90 degrees
        dec = lat - np.pi/2.0
        # Flip ra from latitude to RA (increasing eastward rather than westward)
        ra = -lon % (np.pi*2)
        return ra, dec  
    
    def plotSkyMap(self, metricValueIn, xlabel=None, title='',
                   logScale=False, cbarFormat='%.2g', cmap=cm.jet,
                   percentileClip=None, plotMin=None, plotMax=None,
                   plotMaskedValues=False, zp=None, normVal=None,
                   cbar_edge=True, **kwargs):
        """Plot the sky map of metricValue using healpy Mollweide plot.

        metricValue = metric values
        units = units for metric color-bar label
        title = title for plot
        cbarFormat = format for color bar numerals (i.e. '%.2g', etc) (default to matplotlib default)
        plotMaskedValues = ignored, here to be consistent with OpsimFieldSlicer."""
        # Generate a Mollweide full-sky plot.
        norm = None
        if logScale:
            norm = 'log'
        if cmap is None:
            cmap = cm.jet
        if type(cmap) == str:
            cmap = getattr(cm,cmap)
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')
        if zp:
            metricValue = metricValueIn - zp
        elif normVal:
            metricValue = metricValueIn/normVal
        else:
            metricValue = metricValueIn

        if percentileClip:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=percentileClip)
        if plotMin is None and percentileClip:
            plotMin = pcMin
        if plotMax is None and percentileClip:
            plotMax = pcMax
        if (plotMin is not None) and (plotMax is not None):
            clims = [plotMin, plotMax]
        else:
            clims = None
            
        if clims is not None:
            hp.mollview(metricValue.filled(self.badval), title=title, cbar=False,
                        min=clims[0], max=clims[1], rot=(0,0,180), flip='astro',
                        cmap=cmap, norm=norm)
        else:
            hp.mollview(metricValue.filled(self.badval), title=title, cbar=False,
                        rot=(0,0,180), flip='astro', cmap=cmap, norm=norm)
        hp.graticule(dpar=20., dmer=20.)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        ax = plt.gca()
        im = ax.get_images()[0]
        cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                          extend='both', format=cbarFormat)
        cb.set_label(xlabel)
        # If outputing to PDF, this fixes the colorbar white stripes
        if cbar_edge:
            cb.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number

    def plotHistogram(self, metricValue, title=None, xlabel=None,
                      ylabel='Area (1000s of square degrees)',
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      bins=None, cumulative=False, xMin=None, xMax=None, logScale=False, flipXaxis=False,
                      scale=None, color='b', linestyle='-', **kwargs):
        """Histogram metricValue over the healpix bin points.

        If scale is None, sets 'scale' by the healpix area per slicepoint.
        title = the title for the plot (default None)
        xlabel = x axis label (default None)
        ylabel = y axis label (default 'Area (1000's of square degrees))**
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        bins = bins for histogram (numpy array or # of bins) (default None, uses Freedman-Diaconis rule to set binsize)
        cumulative = make histogram cumulative (default False)
        xMin/Max = histogram range (default None, set by matplotlib hist)
        logScale = use log for y axis (default False)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)."""
        # Simply overrides scale of base plotHistogram. 
        if scale is None:
            scale = (hp.nside2pixarea(self.nside, degrees=True)  / 1000.0)
        fignum = super(HealpixSlicer, self).plotHistogram(metricValue, xlabel=xlabel, ylabel=ylabel,
                                                        title=title, fignum=fignum, 
                                                        label=label, 
                                                        addLegend=addLegend, legendloc=legendloc,
                                                        bins=bins, cumulative=cumulative,
                                                        xMin=xMin, xMax=xMax, logScale=logScale,
                                                        flipXaxis=flipXaxis,
                                                        scale=scale, color=color, linestyle=linestyle,**kwargs)
        return fignum

    def plotPowerSpectrum(self, metricValue, title=None, fignum=None, maxl=500., 
                          label=None, addLegend=False, removeDipole=True, verbose=False, **kwargs):
        """Generate and plot the power spectrum of metricValue.

        maxl = maximum ell value to plot (default 500 .. to plot all l, set to value > 3500)
        title = plot Title (default None)
        fignum = figure number (default None and create new plot)
        label = label to add in figure legend (default None)
        addLegend = flag to add legend (default False).
        removeDipole = remove dipole when calculating power spectrum (default True) (monopole removed automatically.)
        """
        if fignum:
            fig = plt.figure(fignum)
        else:
            fig = plt.figure()
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
        plt.plot(l[condition], cl[condition]*l[condition]*(l[condition]+1), label=label)
        if cl[condition].max() > 0:
            plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l$')
        if addLegend:
            plt.legend(loc='upper right', fancybox=True, prop={'size':'smaller'})
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse/add onto/save this figure if desired). 
        return fig.number


