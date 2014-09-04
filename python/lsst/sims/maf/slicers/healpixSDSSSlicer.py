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
import matplotlib as mpl

class HealpixSDSSSlicer(HealpixSlicer):
    """For use with SDSS stripe 82 square images """
    def __init__(self, nside=128, spatialkey1 ='RA1' , spatialkey2='Dec1', verbose=True, 
                 useCache=True, radius=17./60., leafsize=100, **kwargs):
        """Using one corner of the chip as the spatial key and the diagonal as the radius.  """
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
            # Loop through all the images and check if the slicepoint is inside the corners of the chip
            # XXX--should check if there's a better/faster way to do this.
            # Maybe in the setupSlicer loop through each image, and use the contains_points method to test all the
            # healpixels simultaneously?  Then just have a dict with keys = healpix id and values = list of indices?
            # That way _sliceSimData is just doing a dict look-up and we can get rid of the spatialkey kwargs.

            #wait, this is probably screwing up around zero! 
            indices=[]
            for ind in initIndices:
                bbPath = mplPath.Path( np.array([ [self.corners['RA1'][ind], self.corners['Dec1'][ind]],
                                                 [self.corners['RA2'][ind], self.corners['Dec2'][ind]],
                                                 [self.corners['RA3'][ind], self.corners['Dec3'][ind]],
                                                 [self.corners['RA4'][ind], self.corners['Dec4'][ind]],
                                                 [self.corners['RA1'][ind], self.corners['Dec1'][ind]] ]
                                                 ))
                if bbPath.contains_point((self.slicePoints['ra'][islice],self.slicePoints['dec'][islice])) == 1:
                    indices.append(ind)
            return {'idxs':indices,
                    'slicePoint':{'sid':self.slicePoints['sid'][islice],
                                  'ra':self.slicePoints['ra'][islice],
                                  'dec':self.slicePoints['dec'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)    


    def plotSkyMap(self, metricValueIn, xlabel=None, title='', raMin=-90, raMax=90,
                   raLen=45., decMin=-2., decMax=2.,
                   logScale=False, cbarFormat='%.2f', cmap=cm.jet,
                   percentileClip=None, colorMin=None, colorMax=None,
                   plotMaskedValues=False, zp=None, normVal=None,
                   cbar_edge=True, label=None, **kwargs):
        """
        Plot the sky map of metricValue using healpy cartview plots in thin strips.
        raMin: Minimum RA to plot (deg)
        raMax: Max RA to plot (deg).  Note raMin/raMax define the centers that will be plotted.
        raLen:  Length of the plotted strips in degrees
        decMin: minimum dec value to plot
        decMax: max dec value to plot
        
        metricValueIn: metric values
        units: units for metric color-bar label
        title: title for plot
        cbarFormat: format for color bar numerals (i.e. '%.2g', etc) (default to matplotlib default)
        plotMaskedValues: ignored, here to be consistent with OpsimFieldSlicer."""
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
        racenters=np.arange(raMin,raMax,raLen)
        nframes = racenters.size
        for i, racenter in enumerate(racenters):
            if i == 0:
                useTitle = title +' /n'+'%i < RA < %i'%(racenter-raLen, racenter+raLen)
            else:
                useTitle = '%i < RA < %i'%(racenter-raLen, racenter+raLen)
            hp.cartview(metricValue.filled(self.badval), title=useTitle, cbar=False,
                        min=clims[0], max=clims[1], flip='astro', rot=(racenter,0,0), 
                        cmap=cmap, norm=norm, lonra=[-raLen,raLen],
                        latra=[decMin,decMax], sub=(nframes+1,1,i+1))   
            hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        fig = plt.gcf() 
        ax1 = fig.add_axes([0.1, .15,.8,.075]) #left, bottom, width, height
        # Add label.
        if label is not None:
            plt.figtext(0.8, 0.9, '%s' %label)
        # Make the colorbar as a seperate figure,
        # from http://matplotlib.org/examples/api/colorbar_only.html
        cnorm = colors.Normalize(vmin=clims[0], vmax=clims[1])
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=cnorm, orientation='horizontal')
            cb1.set_label(xlabel)
        # If outputing to PDF, this fixes the colorbar white stripes
        if cbar_edge:
            cb1.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number
