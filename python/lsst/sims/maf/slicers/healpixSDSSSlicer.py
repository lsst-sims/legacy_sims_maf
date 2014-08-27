import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from lsst.sims.maf.utils import percentileClipping
from .healpixSlicer import HealpixSlicer
from functools import wraps
import matplotlib.path as mplPath
import warnings

class HealpixSDSSSlicer(HealpixSlicer):
    """For use with SDSS stripe 82 square images """
    def __init__(self, nside=128, spatialkey1 ='RA1' , spatialkey2='Dec1', verbose=True, 
                 useCache=True, radius=17./60., leafsize=100, **kwargs):
        super(HealpixSDSSSlicer,self).__init__(verbose=verbose,
                                            spatialkey1=spatialkey1, spatialkey2=spatialkey2,
                                            radius=radius, leafsize=leafsize,
                                            useCache=useCache,nside=nside )
        self.cornerLables = ['RA1', 'Dec1', 'RA2','Dec2','RA3','Dec3','RA4','Dec4']

    def setupSlicer(self, simData):
        """Use simData[self.spatialkey1] and simData[self.spatialkey2]
        (in radians) to set up KDTree."""
        self._buildTree(simData[self.spatialkey1], simData[self.spatialkey2], self.leafsize)
        self._setRad(self.radius)
        self.corners = simData[self.cornerLables]
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=spatialkey1/spatialkey2 value .. usually ra/dec)."""
            sx, sy, sz = self._treexyz(self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            # Query against tree.
            initIndices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
            # Loop through all the nearby images and check if the slicepoint is inside the corners of the chip
            # XXX--should check if there's a faster way to do this.
            #isInside = []
            indices=[]
            for ind in initIndices:
                bbPath = mplPath.Path( np.array([ [self.corners['RA1'][ind], self.corners['Dec1'][ind]],
                                                 [self.corners['RA2'][ind], self.corners['Dec2'][ind]],
                                                 [self.corners['RA3'][ind], self.corners['Dec3'][ind]],
                                                 [self.corners['RA4'][ind], self.corners['Dec4'][ind]],
                                                 [self.corners['RA1'][ind], self.corners['Dec1'][ind]] ]
                                                 ))
                #isInside.append(bbPath.contains_point((self.slicePoints['ra'][islice],
                #                                      self.slicePoints['dec'][islice]) ))
                if bbPath.contains_point((self.slicePoints['ra'][islice],self.slicePoints['dec'][islice])) == 1:
                    indices.append(ind)
                
            #insideInd = [indices[i] for i,val in enumerate(isInside) if val == 1 ]
            #indices = insideInd
            #else:
            #    indices = []
            return {'idxs':indices,
                    'slicePoint':{'sid':self.slicePoints['sid'][islice],
                                  'ra':self.slicePoints['ra'][islice],
                                  'dec':self.slicePoints['dec'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)    


    def plotSkyMap(self, metricValueIn, xlabel=None, title='',
                   logScale=False, cbarFormat='%.2f', cmap=cm.jet,
                   percentileClip=None, colorMin=None, colorMax=None,
                   plotMaskedValues=False, zp=None, normVal=None,
                   cbar_edge=True, label=None, latra=[-5,5], lonra=[-100,100],
                   **kwargs):
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
        if colorMin is None and percentileClip:
            colorMin = pcMin
        if colorMax is None and percentileClip:
            colorMax = pcMax
        if (colorMin is not None) or (colorMax is not None):
            clims = [colorMin, colorMax]
        else:
            clims = None

        # Make sure there is some range on the colorbar
        if clims is None:
            if metricValue.compressed().size > 0:
                clims=[metricValue.compressed().min(), metricValue.compressed().max()]
            else:
                clims = [-1,1]
            if clims[0] == clims[1]:
                clims[0] =  clims[0]-1
                clims[1] =  clims[1]+1        
                   
        hp.cartview(metricValue.filled(self.badval), title=title, cbar=False,
                    min=clims[0], max=clims[1], rot=(0,0,180), flip='astro',
                    cmap=cmap, norm=norm, latra=latra, lonra=lonra)        
        hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        ax = plt.gca()
        im = ax.get_images()[0]
        # Add label.
        if label is not None:
            plt.figtext(0.8, 0.9, '%s' %label)
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                              extend='both', extendrect=True, format=cbarFormat)
            cb.set_label(xlabel)
        # If outputing to PDF, this fixes the colorbar white stripes
        if cbar_edge:
            cb.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number
