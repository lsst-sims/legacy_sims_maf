# The base class for all spatial slicers.
# Slicers are 'data slicers' at heart; spatial slicers slice data by RA/Dec and 
#  return the relevant indices in the simData to the metric. 
# The primary things added here are the methods to slice the data (for any spatial slicer)
#  as this uses a KD-tree built on spatial (RA/Dec type) indexes. 

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse   
from matplotlib.ticker import FuncFormatter
from functools import wraps
import warnings
from lsst.sims.maf.utils import optimalBins, percentileClipping

try:
    # Try cKDTree first, as it's supposed to be faster.
    from scipy.spatial import cKDTree as kdtree
    #current stack scipy has a bad version of cKDTree.  
    if not hasattr(kdtree,'query_ball_point'): 
        from scipy.spatial import KDTree as kdtree
except:
    # But older scipy may not have cKDTree.
    from scipy.spatial import KDTree as kdtree

from .baseSlicer import BaseSlicer

class BaseSpatialSlicer(BaseSlicer):
    """Base slicer object, with added slicing functions for spatial slicer."""
    def __init__(self, verbose=True, spatialkey1='fieldRA', spatialkey2='fieldDec', 
                 badval=-666, leafsize=100, radius=1.75):
        """Instantiate the base spatial slicer object.
        spatialkey1 = ra, spatialkey2 = dec, typically.
        'leafsize' is the number of RA/Dec pointings in each leaf node of KDtree
        'radius' (in degrees) is distance at which matches between
        the simData KDtree 
        and slicePoint RA/Dec values will be produced."""
        super(BaseSpatialSlicer, self).__init__(verbose=verbose, badval=badval)
        self.spatialkey1 = spatialkey1
        self.spatialkey2 = spatialkey2
        self.columnsNeeded = [spatialkey1, spatialkey2]
        self.slicer_init={'spatialkey1':spatialkey1, 'spatialkey2':spatialkey2,
                          'radius':radius, 'badval':badval}
        self.radius = radius
        self.leafsize=leafsize
        # RA and Dec are required slicePoint info for any spatial slicer.
        self.slicePoints['sid'] = None
        self.slicePoints['ra'] = None
        self.slicePoints['dec'] = None
        self.nslice = None

    def setupSlicer(self, simData):
        """Use simData[self.spatialkey1] and simData[self.spatialkey2]
        (in radians) to set up KDTree."""
        self._buildTree(simData[self.spatialkey1], simData[self.spatialkey2], self.leafsize)
        self._setRad(self.radius)
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=spatialkey1/spatialkey2 value .. usually ra/dec)."""
            sx, sy, sz = self._treexyz(self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            # Query against tree.
            indices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
            return {'idxs':indices,
                    'slicePoint':{'sid':self.slicePoints['sid'][islice],
                                  'ra':self.slicePoints['ra'][islice],
                                  'dec':self.slicePoints['dec'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)    

    def _treexyz(self, ra, dec):
        """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
        # Note ra/dec can be arrays.
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z
    
    def _buildTree(self, simDataRa, simDataDec, 
                  leafsize=100):
        """Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

        simDataRA, simDataDec = RA and Dec values (in radians).
        leafsize = the number of Ra/Dec pointings in each leaf node."""
        if np.any(np.abs(simDataRa) > np.pi*2.0) or np.any(np.abs(simDataDec) > np.pi*2.0):
            raise ValueError('Expecting RA and Dec values to be in radians.')
        x, y, z = self._treexyz(simDataRa, simDataDec)
        data = zip(x,y,z)
        if np.size(data) > 0:
            self.opsimtree = kdtree(data, leafsize=leafsize)
        else:
            raise ValueError('SimDataRA and Dec should have length greater than 0.')

    def _setRad(self, radius=1.75):
        """Set radius (in degrees) for kdtree search.
        
        kdtree queries will return pointings within rad."""        
        x0, y0, z0 = (1, 0, 0)
        x1, y1, z1 = self._treexyz(np.radians(radius), 0)
        self.rad = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    
    def sliceSimDataMultiSlicepoint(self, islices):
        """Return indexes for opsim data at multiple slicepoints (rarely used). """
        binx, biny, binz=self._treexyz(self.slicePoints['ra'][islices], self.slicePoints['dec'][islices])
        indices = self.opsimtree.query_ball_point(zip(binx, biny, binz), self.rad)
        return indices

        
    ## Plot histogram (base spatial slicer method).
    def plotHistogram(self, metricValueIn, title=None, xlabel=None, units=None, ylabel=None,
                      fignum=None, label=None, addLegend=False, legendloc='upper left',
                      bins=None, cumulative=False, xMin=None, xMax=None, yMin=None, yMax=None,
                      logScale='auto', flipXaxis=False,
                      scale=1.0, yaxisformat='%.3f', color='b',
                      zp=None, normVal=None, percentileClip=None, **kwargs):
        """Plot a histogram of metricValue, labelled by metricLabel.

        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        label = the label to use in the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        bins = bins for histogram (numpy array or # of bins)
        (default None, uses Freedman-Diaconis rule to set binsize)
        cumulative = make histogram cumulative (default False)
        xMin/Max = histogram range (default None, set by matplotlib hist)
        yMin/Max = histogram y range
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)
        zp = zeropoing to subtract off metricVals
        normVal = normalization value to divide metric values by (overrides zp)"""
        if bins is None:
            bins = optimalBins(metricValueIn)
        # Histogram metricValues. 
        fig = plt.figure(fignum)
        if not xlabel:
            xlabel = units
        if zp:
            metricValue = metricValueIn.compressed() - zp
        elif normVal:
            metricValue = metricValueIn.compressed()/normVal
        else:
            metricValue = metricValueIn.compressed()
        # Need to only use 'good' values in histogram,
        # but metricValue is masked array (so bad values masked when calculating max/min).
        if xMin is None and xMax is None:
            if percentileClip:
                xMin, xMax = percentileClipping(metricValue, percentile=percentileClip)
                histRange = [xMin, xMax]
            else:
                histRange = None
        else:
            histRange=[xMin, xMax]
        if yMin is not None or yMax is not None:
            plt.ylim([yMin,yMax])
        # See if should use log scale.
        if logScale == 'auto':
            if (np.log10(np.max(histRange)-np.min(histRange)) > 3 ) & (np.min(histRange) > 0):
                logScale = True
            else:
                logScale = False
        # Plot histograms.
        # Add a test to see if data falls within histogram range.. because otherwise histogram will fail.
        if histRange is not None:
            if (histRange[0] is None) and (histRange[1] is not None):
                condition = (metricValue <= histRange[1])
            elif (histRange[1] is None) and (histRange[0] is not None):
                condition = (metricValue >= histRange[0])
            else:
                condition = ((metricValue >= histRange[0]) & (metricValue <= histRange[1]))
            plotValue = metricValue[condition]
        else:
            plotValue = metricValue
        if plotValue.size == 0:
            if histRange is None:
                warnings.warn('Could not plot metric data: histRange is None and all data masked' )
            else:
                warnings.warn('Could not plot metric data: none fall within histRange %.2f %.2f' %
                              (histRange[0], histRange[1]))
            return None
        else:
            n, b, p = plt.hist(plotValue, bins=bins, histtype='step', log=logScale,
                               cumulative=cumulative, range=histRange, label=label, color=color)
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(y,  pos):        
            return yaxisformat % (y * scale)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        # There is a bug in histype='step' that can screw up the ylim.  Comes up when running allSlicer.Cfg.py
        if plt.axis()[2] == max(n):
            plt.ylim([n.min(),n.max()]) 
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if flipXaxis:
            # Might be useful for magnitude scales.
            x0, x1 = plt.xlim()
            plt.xlim(x1, x0)
        if addLegend:
            plt.legend(fancybox=True, prop={'size':'smaller'}, loc=legendloc)
        if title!=None:
            plt.title(title)
        # Return figure number (so we can reuse this if desired).         
        return fig.number
            
    ### Generate sky map (base spatial slicer methods, using ellipses for each RA/Dec value)
    ### a healpix slicer will not have self.ra / self.dec functions, but plotSkyMap is overriden.
    
    def _plot_tissot_ellipse(self, lon, lat, radius, ax=None, **kwargs):
        """Plot Tissot Ellipse/Tissot Indicatrix
        
        Parameters
        ----------
        lon : float or array_like
        longitude-like of ellipse centers (radians)
        lat : float or array_like
        latitude-like of ellipse centers (radians)
        radius : float or array_like
        radius of ellipses (radians)
        ax : Axes object (optional)
        matplotlib axes instance on which to draw ellipses.
        
        Other Parameters
        ----------------
        other keyword arguments will be passed to matplotlib.patches.Ellipse.

        # The code in this method adapted from astroML, which is BSD-licensed. 
        # See http://github.com/astroML/astroML for details.
        """
        # Code adapted from astroML, which is BSD-licensed. 
        # See http://github.com/astroML/astroML for details.
        ellipses = []
        if ax is None:
            ax = plt.gca()            
        for l, b, diam in np.broadcast(lon, lat, radius*2.0):
            el = Ellipse((l, b), diam / np.cos(b), diam)
            ellipses.append(el)
        return ellipses

    def _plot_ecliptic(self, ax=None):
        """Plot a red line at location of ecliptic"""
        if ax is None:
            ax = plt.gca()
        ecinc = 23.439291*(np.pi/180.0)
        x_ec = np.arange(0, np.pi*2., (np.pi*2./360))
        ra = x_ec - np.pi
        y_ec = np.sin(x_ec) * ecinc
        plt.plot(ra, y_ec, 'r-')        
        
    def plotSkyMap(self, metricValueIn, title=None, xlabel=None, units=None,
                   projection='aitoff', radius=1.75/180.*np.pi,
                   logScale='auto', cbarFormat=None, cmap=cm.jet, fignum=None,
                   plotMaskedValues=False, zp=None, normVal=None,
                   colorMin=None, colorMax=None, percentileClip=None,  cbar_edge=True, **kwargs):
        """Plot the sky map of metricValue."""
        from matplotlib.collections import PatchCollection
        from matplotlib import colors
        if fignum is None:
            fig = plt.figure()
        if zp or normVal:
            if zp:
                metricValue = metricValueIn - zp
            if normVal:
                metricValue = metricValueIn/normVal
        else:
            metricValue = metricValueIn
        # other projections available include 
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        ax = plt.subplot(111,projection=projection)
        # Only include points that are not masked
        if plotMaskedValues:
            goodPts = np.arange(metricValue.size)
        else:
            goodPts = np.where(metricValue.mask == False)[0]
        # Add points for RA/Dec locations
        lon = -(self.slicePoints['ra'][goodPts] - np.pi) % (np.pi*2) - np.pi
        ellipses = self._plot_tissot_ellipse(lon, self.slicePoints['dec'][goodPts], radius, ax=ax)
        if logScale == 'auto':
            if ((np.log10(np.max(metricValue[goodPts])-np.min(metricValue[goodPts])) > 3 ) &
                (np.min(metricValue[goodPts]) > 0)):
                logScale = True
            else:
                logScale = False
        if logScale:
            norml = colors.LogNorm()
            p = PatchCollection(ellipses, cmap=cmap, alpha=1, linewidth=0, edgecolor=None,
                                norm=norml, rasterized=True)
        else:
            p = PatchCollection(ellipses, cmap=cmap, alpha=1, linewidth=0, edgecolor=None, rasterized=True)
        p.set_array(metricValue.filled(self.badval)[goodPts])
        ax.add_collection(p)
        # Add ecliptic
        self._plot_ecliptic(ax=ax)
        ax.grid(True)
        ax.xaxis.set_ticklabels([])
        # Add color bar (with optional setting of limits)
        if percentileClip:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=percentileClip)
        if colorMin is None and percentileClip:
            colorMin = pcMin
        else:
            colorMin = metricValue.compressed().min()
        if colorMax is None and percentileClip:
            colorMax = pcMax
        else:
            colorMax = metricValue.compressed().max()
        # Combine to make clims:
        clims = [colorMin, colorMax]
        p.set_clim(clims)
        cb = plt.colorbar(p, aspect=25, extend='both', orientation='horizontal', format=cbarFormat)
        # If outputing to PDF, this fixes the colorbar white stripes
        if cbar_edge:
            cb.solids.set_edgecolor("face")
        if xlabel is not None:
            cb.set_label(xlabel)
        elif units is not None:
            cb.set_label(units)
        if title is not None:
            plt.text(0.5, 1.09, title, horizontalalignment='center', transform=ax.transAxes)
        return fig.number
