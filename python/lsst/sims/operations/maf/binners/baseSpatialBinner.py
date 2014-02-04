# The base class for all spatial binners.
# Binners are 'data slicers' at heart; spatial binners slice data by RA/Dec and 
#  return the relevant indices in the simData to the metric. 
# The primary things added here are the methods to slice the data (for any spatial binner)
#  as this uses a KD-tree built on spatial (RA/Dec type) indexes. 

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse   
from matplotlib.ticker import FuncFormatter

try:
    # Try cKDTree first, as it's supposed to be faster.
    from scipy.spatial import cKDTree as kdtree
    #current stack scipy has a bad version of cKDTree.  
    if not hasattr(kdtree,'query_ball_point'): 
        from scipy.spatial import KDTree as kdtree
except:
    # But older scipy may not have cKDTree.
    from scipy.spatial import KDTree as kdtree

from .baseBinner import BaseBinner

class BaseSpatialBinner(BaseBinner):
    """Base binner object, with added slicing functions for spatial binner."""
    def __init__(self, verbose=True, *args, **kwargs):
        """Instantiate the base spatial binner object."""
        super(BaseSpatialBinner, self).__init__(verbose=verbose)
        self.binnertype = 'SPATIAL'

    def setupBinner(self, simData, spatialkey1,
                    spatialkey2, leafsize=100, radius=1.8):
        """Use simData['spatialkey1'] and simData['spatialkey2']
        (in radians) to set up KDTree. 

        'leafsize' is the number of RA/Dec pointings in each leaf node of KDtree
        'radius' (in degrees) is distance at which matches between
        the simData KDtree 
        and binpoint RA/Dec values will be produced."""
        self._buildTree(simData[spatialkey1], simData[spatialkey2], leafsize)
        self._setRad(radius)    
    
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
        if np.any(simDataRa > np.pi*2.0) or np.any(simDataDec> np.pi*2.0):
            raise Exception('Expecting RA and Dec values to be in radians.')
        x, y, z = self._treexyz(simDataRa, simDataDec)
        data = zip(x,y,z)
        self.opsimtree = kdtree(data, leafsize=leafsize)

    def _setRad(self, radius=1.8):
        """Set radius (in degrees) for kdtree search.
        
        kdtree queries will return pointings within rad."""        
        x0, y0, z0 = (1, 0, 0)
        x1, y1, z1 = self._treexyz(np.radians(radius), 0)
        self.rad = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    
    def sliceSimData(self, binpoint):
        """Return indexes for relevant opsim data at binpoint (binpoint=ra/dec value)."""
        binx, biny, binz = self._treexyz(binpoint[0], binpoint[1])
        # If we were given more than one binpoint, try multiple query against the tree.
        if isinstance(binx, np.ndarray):
            indices = self.opsimtree.query_ball_point(zip(binx, biny, binz), 
                                                      self.rad)
        # If we were given one binpoint, do a single query against the tree.
        else:
            indices = self.opsimtree.query_ball_point((binx, biny, binz), 
                                                      self.rad)
        return indices

    ## Plot histogram (base spatial binner method).
        
    def plotHistogram(self, metricValue, metricLabel, title=None, 
                      fignum=None, legendLabel=None, addLegend=False, legendloc='upper left',
                      bins=100, cumulative=False, histRange=None, flipXaxis=False,
                      scale=1.0, yaxisformat='%.3f'):
        """Plot a histogram of metricValue, labelled by metricLabel.

        title = the title for the plot (default None)
        fignum = the figure number to use (default None - will generate new figure)
        legendLabel = the label to use for the figure legend (default None)
        addLegend = flag for whether or not to add a legend (default False)
        legendloc = location for legend (default 'upper left')
        bins = bins for histogram (numpy array or # of bins) (default 100)
        cumulative = make histogram cumulative (default False)
        histRange = histogram range (default None, set by matplotlib hist)
        flipXaxis = flip the x axis (i.e. for magnitudes) (default False)
        scale = scale y axis by 'scale' (i.e. to translate to area)"""
        # Histogram metricValues. 
        fig = plt.figure(fignum)
        # Need to only use 'good' values in histogram.
        good = np.where(metricValue != self.badval)
        if metricValue[good].min() >= metricValue[good].max():
            if histRange==None:
                histRange = [metricValue[good].min() , metricValue[good].min() + 1]
                raise warnings.warn('Max (%f) of metric Values was less than or equal to min (%f). Using (min value/min value + 1) as a backup for histRange.' 
                                    % (metricValue[good].max(), metricValue[good].min()))
        n, b, p = plt.hist(metricValue[good], bins=bins, histtype='step', 
                           cumulative=cumulative, range=histRange, label=legendLabel)        
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(x,  pos):        
            return yaxisformat % (x * scale)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        plt.xlabel(metricLabel)
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
            
    ### Generate sky map (base spatial binner methods, using ellipses for each RA/Dec value)
    ### a healpix binner will not have self.ra / self.dec functions, but plotSkyMap is overriden.
    
    def _plot_tissot_ellipse(self, longitude, latitude, radius, ax=None, **kwargs):
        """Plot Tissot Ellipse/Tissot Indicatrix
        
        Parameters
        ----------
        longitude : float or array_like
        longitude of ellipse centers (radians)
        latitude : float or array_like
        latitude of ellipse centers (radians)
        radius : float or array_like
        radius of ellipses
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
        for long, lat, rad in np.broadcast(longitude, latitude, radius*2.0):
            el = Ellipse((long, lat), rad / np.cos(lat), rad)
            ellipses.append(el)
        return ellipses

        
    def plotSkyMap(self, metricValue, metricLabel, title=None, projection='aitoff',
                   clims=None, cbarFormat='%.2g', cmap=cm.jet, fignum=None):
        """Plot the sky map of metricValue."""
        from matplotlib.collections import PatchCollection
        if fignum==None:
            fig = plt.figure()
        ax = plt.subplot(projection=projection)        
        # other projections available include 
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        radius = 1.8 * np.pi / 180.
        ellipses = self._plot_tissot_ellipse((self.ra - np.pi), self.dec, radius, ax=ax)
        p = PatchCollection(ellipses, cmap=cmap, alpha=1, linewidth=0, edgecolor=None)
        p.set_array(metricValue)
        ax.add_collection(p)
        if clims != None:
            p.set_clim(clims)
        cb = plt.colorbar(p, orientation='horizontal', format=cbarFormat)
        if title != None:
            plt.text(0.5, 1.09, title, horizontalalignment='center', transform=ax.transAxes)
        ax.grid()
        return fig.number
