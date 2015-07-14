import numpy as np
import warnings
import healpy as hp
from matplotlib import colors
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
from matplotlib.patches import Ellipse
from matplotlib.collections import PatchCollection

from lsst.sims.maf.utils import optimalBins, percentileClipping
from .plotHandler import BasePlotter

from lsst.sims.utils import equatorialFromGalactic
import inspect

__all__ = ['HealpixSkyMap', 'HealpixPowerSpectrum', 'HealpixHistogram', 'OpsimHistogram',
           'BaseHistogram', 'BaseSkyMap', 'HealpixSDSSSkyMap']


class HealpixSkyMap(BasePlotter):
    def __init__(self):
        # Set the plotType
        self.plotType = 'SkyMap'
        self.objectPlotter = False
        # Set up the default plotting parameters.
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'logScale':False, 'cbarFormat':None, 'cmap':cm.jet,
                                'percentileClip':None, 'colorMin':None, 'colorMax':None,
                                'zp':None, 'normVal':None,
                                'cbar_edge':True, 'nTicks':None, 'rot':(0,0,0)}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Generate a sky map of healpix metric values using healpy's mollweide view.
        """
        # Check that the slicer is a HealpixSlicer, or subclass thereof
        # Using the names rather than just comparing the classes themselves
        # to avoid circular dependency between slicers and plots
        classes = inspect.getmro(slicer.__class__)
        cnames = [cls.__name__ for cls in classes]
        if 'HealpixSlicer' not in cnames:
            raise ValueError('HealpixSkyMap is for use with healpix slicers')
        fig = plt.figure(fignum)
        # Override the default plotting parameters with user specified values.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        # Generate a Mollweide full-sky plot.
        norm = None
        if plotDict['logScale']:
            norm = 'log'
        cmap = plotDict['cmap']
        if type(cmap) == str:
            cmap = getattr(cm, cmap)
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')
        if plotDict['zp']:
            metricValue = metricValueIn - plotDict['zp']
        elif plotDict['normVal']:
            metricValue = metricValueIn/plotDict['normVal']
        else:
            metricValue = metricValueIn
        # Set up color bar limits.
        if plotDict['percentileClip']:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=plotDict['percentileClip'])
        colorMin = plotDict['colorMin']
        colorMax = plotDict['colorMax']
        if colorMin is None and plotDict['percentileClip']:
            colorMin = pcMin
        if colorMax is None and plotDict['percentileClip']:
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
                clims = [-1, 1]
            if clims[0] == clims[1]:
                clims[0] =  clims[0] - 1
                clims[1] =  clims[1] + 1
        hp.mollview(metricValue.filled(slicer.badval), title=plotDict['title'], cbar=False,
                    min=clims[0], max=clims[1], rot=plotDict['rot'], flip='astro',
                    cmap=cmap, norm=norm, fig=fig.number)
        # This graticule call can fail with old versions of healpy and matplotlib 1.4.0.
        # Make sure the latest version of healpy in the stack is setup
        hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because we want more tickmarks).
        ax = plt.gca()
        im = ax.get_images()[0]
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.8, 0.8, '%s' %(plotDict['label']))
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = plt.colorbar(im, shrink=0.75, aspect=25, orientation='horizontal',
                            extend='both', extendrect=True, format=plotDict['cbarFormat'])
            cb.set_label(plotDict['xlabel'])
            if plotDict['nTicks'] is not None:
                tick_locator = ticker.MaxNLocator(nbins=plotDict['nTicks'])
                cb.locator = tick_locator
                cb.update_ticks()
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb.solids.set_edgecolor("face")
        return fig.number

class HealpixPowerSpectrum(BasePlotter):
    def __init__(self):
        self.plotType = 'PowerSpectrum'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'label':None,
                                'maxl':None, 'removeDipole':True,
                                'logScale':True, 'color':'b', 'linestyle':'-'}

    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Generate and plot the power spectrum of metricValue (calculated on a healpix grid).
        """
        if slicer.slicerName != 'HealpixSlicer':
            raise ValueError('HealpixPowerSpectrum for use with healpix metricBundles.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        fig = plt.figure(fignum)
        # If the mask is True everywhere (no data), just plot zeros
        if False not in metricValue.mask:
            return None
        if plotDict['removeDipole']:
            cl = hp.anafast(hp.remove_dipole(metricValue.filled(slicer.badval)), lmax=plotDict['maxl'])
        else:
            cl = hp.anafast(metricValue.filled(slicer.badval), lmax=plotDict['maxl'])
        ell = np.arange(np.size(cl))
        if plotDict['removeDipole']:
            condition = (ell > 1)
        else:
            condition = (ell > 0)
        ell = ell[condition]
        cl = cl[condition]
        # Plot the results.
        plt.plot(ell, (cl*ell*(ell+1))/2.0/np.pi,
                 color=plotDict['color'], linestyle=plotDict['linestyle'], label=plotDict['label'])
        if cl.max() > 0 and plotDict['logScale']:
            plt.yscale('log')
        plt.xlabel(r'$l$')
        plt.ylabel(r'$l(l+1)C_l/(2\pi)$')
        if plotDict['title'] is not None:
            plt.title(plotDict['title'])
        # Return figure number (so we can reuse/add onto/save this figure if desired).
        return fig.number

class HealpixHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None,
                                'ylabel':'Area (1000s of square degrees)', 'label':None,
                                'bins':None, 'binsize':None, 'cumulative':False,
                                'scale':None, 'xMin':None, 'xMax':None,
                                'logScale':False, 'color':'b'}
        self.baseHist = BaseHistogram()
    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Histogram metricValue for all healpix points.
        """
        if slicer.slicerName != 'HealpixSlicer':
            raise ValueError('HealpixHistogram is for use with healpix slicer.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if plotDict['scale'] is None:
            plotDict['scale'] = (hp.nside2pixarea(slicer.nside, degrees=True)  / 1000.0)
        fignum = self.baseHist(metricValue, slicer, plotDict, fignum=fignum)
        return fignum

class OpsimHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'ylabel':'Number of Fields', 'yaxisFormat':'%d',
                                'bins':None, 'binsize':None, 'cumulative':False,
                                'scale':1.0, 'xMin':None, 'xMax':None,
                                'logScale':False, 'color':'b'}
        self.baseHist = BaseHistogram()
    def __call__(self, metricValue, slicer, userPlotDict, fignum=None):
        """
        Histogram metricValue for all healpix points.
        """
        if slicer.slicerName != 'OpsimFieldSlicer':
            raise ValueError('OpsimHistogram is for use with OpsimFieldSlicer.')
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        fignum = self.baseHist(metricValue, slicer, plotDict, fignum=fignum)
        return fignum

class BaseHistogram(BasePlotter):
    def __init__(self):
        self.plotType = 'Histogram'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'ylabel':'Count', 'label':None,
                                'bins':None, 'binsize':None, 'cumulative':False,
                                'scale':1.0, 'xMin':None, 'xMax':None,
                                'logScale':'auto', 'color':'b',
                                'yaxisformat':'%.3f',
                                'zp':None, 'normVal':None, 'percentileClip':None}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Plot a histogram of metricValues (such as would come from a spatial slicer).
        """
        plotType = 'Histogram'
        # Adjust metric values by zeropoint or normVal, and use 'compressed' version of masked array.
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)
        if plotDict['zp']:
            metricValue = metricValueIn.compressed() - plotDict['zp']
        elif plotDict['normVal']:
            metricValue = metricValueIn.compressed()/plotDict['normVal']
        else:
            metricValue = metricValueIn.compressed()
        # Determine percentile clipped X range, if set. (and xmin/max not set).
        if plotDict['xMin'] is None and plotDict['xMax'] is None:
            if plotDict['percentileClip']:
                plotDict['xMin'], plotDict['xMax'] = percentileClipping(metricValue, percentile=plotDict['percentileClip'])
        # Determine range for histogram. Note that if xmin/max are None, this will just be [None, None].
        histRange = [plotDict['xMin'], plotDict['xMax']]
        # Should we use log scale on y axis? (if 'auto')
        if plotDict['logScale'] == 'auto':
            plotDict['logScale'] = False
            if np.min(histRange) > 0:
                if (np.log10(np.max(histRange)-np.log10(np.min(histRange))) > 3 ):
                    plotDict['logScale'] = True
        # If binsize was specified, set up an array of bins for the histogram.
        if plotDict['binsize'] is not None:
            #  If generating cumulative histogram, want to use full range of data (but with given binsize).
            #    .. but if user set histRange to be wider than full range of data, then
            #       extend bins to cover this range, so we can make prettier plots.
            if plotDict['cumulative'] is not False:
                if histRange[0] is not None:
                    bmin = np.min([metricValue.min(), histRange[0]])
                else:
                    bmin = metricValue.min()
                if histRange[1] is not None:
                    bmax = np.max([metricValue.max(), histRange[1]])
                else:
                    bmax = metricValue.max()
                bins = np.arange(bmin, bmax+plotDict['binsize']/2.0, plotDict['binsize'])
            #  Else try to set up bins using min/max values if specified, or full data range.
            else:
                if histRange[0] is not None:
                    bmin = histRange[0]
                else:
                    bmin = metricValue.min()
                if histRange[1] is not None:
                    bmax = histRange[1]
                else:
                    bmax = metricValue.max()
                bins = np.arange(bmin, bmax+plotDict['binsize']/2.0, plotDict['binsize'])
        # Otherwise, determine number of bins, if neither 'bins' or 'binsize' were specified.
        else:
            if plotDict['bins'] is None:
                bins = optimalBins(metricValue)
            else:
                bins = plotDict['bins']
        # Generate plots.
        fig = plt.figure(fignum)
        if plotDict['cumulative'] is not False:
            # If cumulative is set, generate histogram without using histRange (to use full range of data).
            n, b, p = plt.hist(metricValue, bins=bins, histtype='step', log=plotDict['logScale'],
                                cumulative=plotDict['cumulative'], label=plotDict['label'], color=plotDict['color'])
        else:
            # Plot non-cumulative histogram.
            # First, test if data falls within histRange, because otherwise histogram generation will fail.
            if np.min(histRange) is not None:
                if (histRange[0] is None) and (histRange[1] is not None):
                    condition = (metricValue <= histRange[1])
                elif (histRange[1] is None) and (histRange[0] is not None):
                    condition = (metricValue >= histRange[0])
                else:
                    condition = ((metricValue >= histRange[0]) & (metricValue <= histRange[1]))
                plotValue = metricValue[condition]
            else:
                plotValue = metricValue
            # If there is only one value to histogram, need to set histRange, otherwise histogram will fail.
            rangePad = 20.
            if (np.unique(plotValue).size == 1) & (np.min(histRange) is None):
                warnings.warn('Only one metric value, making a guess at a good histogram range.')
                histRange = [plotValue.min()-rangePad, plotValue.max()+rangePad]
                if (plotValue.min() >= 0) & (histRange[0] < 0):
                    # Reset histogram range if it went below 0.
                    histRange[0] = 0.
                if 'binsize' in plotDict:
                    bins = np.arange(histRange[0], histRange[1], plotDict['binsize'])
                else:
                    bins = np.arange(histRange[0], histRange[1], (histRange[1] - histRange[0])/50.)
            # If there is no data within the histogram range, we will generate an empty plot.
            # If there is data, make the histogram.
            if plotValue.size > 0:
                # Generate histogram.
                if np.min(histRange) is None:
                    histRange = None
                n, b, p = plt.hist(plotValue, bins=bins, histtype='step', log=plotDict['logScale'],
                                    cumulative=plotDict['cumulative'], range=histRange,
                                    label=plotDict['label'], color=plotDict['color'])

        # Fill in axes labels and limits.
        # Option to use 'scale' to turn y axis into area or other value.
        def mjrFormatter(y,  pos):
            try:
                return plotDict['yaxisformat'] % (y * plotDict['scale'])
            except:
                import pdb ; pdb.set_trace()
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FuncFormatter(mjrFormatter))
        # Set y limits.
        if 'yMin' in plotDict:
            if plotDict['yMin'] is not None:
                plt.ylim(ymin=plotDict['yMin'])
        else:
            # There is a bug in histype='step' that can screw up the ylim.  Comes up when running allSlicer.Cfg.py
            try:
                if plt.axis()[2] == max(n):
                    plt.ylim([n.min(),n.max()])
            except UnboundLocalError:
                # This happens if we were generating an empty plot (no histogram).
                # But in which case, the above error was probably not relevant. So skip it.
                pass
        if 'yMax' in plotDict:
            plt.ylim(ymax=plotDict['yMax'])
        # Set x limits.
        if plotDict['xMin'] is not None:
            plt.xlim(xmin=plotDict['xMin'])
        if plotDict['xMax'] is not None:
            plt.xlim(xmax=plotDict['xMax'])
        # Set/Add various labels.
        plt.xlabel(plotDict['xlabel'])
        plt.ylabel(plotDict['ylabel'])
        plt.title(plotDict['title'])
        # Return figure number
        return fig.number


class BaseSkyMap(BasePlotter):
    def __init__(self):
        self.plotType = 'SkyMap'
        self.objectPlotter = False # unless 'metricIsColor' is true..
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'label':None,
                                'projection':'aitoff', 'radius':np.radians(1.75),
                                'logScale':'auto', 'cbar':True, 'cbarFormat':None,
                                'cmap':cm.jet, 'alpha':1.0,
                                'zp':None, 'normVal':None,
                                'colorMin':None, 'colorMax':None, 'percentileClip':False,
                                'cbar_edge':True, 'plotMask':False, 'metricIsColor':False,
                                'raCen':0.0, 'mwZone':True}

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
            el = Ellipse((l, b), diam / np.cos(b), diam, **kwargs)
            ellipses.append(el)
        return ellipses

    def _plot_ecliptic(self, raCen=0, ax=None):
        """
        Plot a red line at location of ecliptic.
        """
        if ax is None:
            ax = plt.gca()
        ecinc = 23.439291*(np.pi/180.0)
        ra_ec = np.arange(0, np.pi*2., (np.pi*2./360))
        dec_ec = np.sin(ra_ec) * ecinc
        lon = -(ra_ec - raCen - np.pi) % (np.pi*2) - np.pi
        ax.plot(lon, dec_ec, 'r.', markersize=1.8, alpha=0.4)

    def _plot_mwZone(self, raCen=0, peakWidth=np.radians(10.), taperLength=np.radians(80.), ax=None):
        """
        Plot blue lines to mark the milky way galactic exclusion zone.
        """
        if ax is None:
            ax = plt.gca()
        # Calculate galactic coordinates for mw location.
        step = 0.02
        galL = np.arange(-np.pi, np.pi+step/2., step)
        val = peakWidth * np.cos(galL/taperLength*np.pi/2.)
        galB1 = np.where(np.abs(galL) <= taperLength, val, 0)
        galB2 = np.where(np.abs(galL) <= taperLength, -val, 0)
        # Convert to ra/dec.
        # Convert to lon/lat and plot.
        ra, dec = equatorialFromGalactic(galL, galB1)
        lon = -(ra - raCen - np.pi) %(np.pi*2) - np.pi
        ax.plot(lon, dec, 'b.', markersize=1.8, alpha=0.4)
        ra, dec = equatorialFromGalactic(galL, galB2)
        lon = -(ra - raCen - np.pi) %(np.pi*2) - np.pi
        ax.plot(lon, dec, 'b.', markersize=1.8, alpha=0.4)

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None):
        """
        Plot the sky map of metricValue for a generic spatial slicer.
        """
        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        metricValue = metricValueIn
        if plotDict['zp'] is not None :
            metricValue = metricValue - plotDict['zp']
        if plotDict['normVal'] is not None:
            metricValue = metricValue/plotDict['normVal']
        # other projections available include
        # ['aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
        ax = fig.add_subplot(111, projection=plotDict['projection'])
        # Set up valid datapoints and colormin/max values.
        if plotDict['plotMask']:
            # Plot all data points.
            mask = np.ones(len(metricValue), dtype='bool')
        else:
            # Only plot points which are not masked. Flip numpy ma mask where 'False' == 'good'.
            mask = ~metricValue.mask
        # Determine color min/max values. metricValue.compressed = non-masked points.
        if plotDict['percentileClip']:
            pcMin, pcMax = percentileClipping(metricValue.compressed(), percentile=plotDict['percentileClip'])
        if plotDict['colorMin'] is None:
            if plotDict['percentileClip']:
                plotDict['colorMin'] = pcMin
            else:
                plotDict['colorMin'] = metricValue.compressed().min()
        if plotDict['colorMax'] is None:
            if plotDict['percentileClip']:
                plotDict['colorMax'] = pcMax
            else:
                plotDict['colorMax'] = metricValue.compressed().max()
                # Avoid colorbars with no range.
                if plotDict['colorMax'] == plotDict['colorMin']:
                    plotDict['colorMax'] = plotDict['colorMax'] + 1
                    plotDict['colorMin'] = plotDict['colorMin'] - 1
        # Combine to make clims:
        clims = [plotDict['colorMin'], plotDict['colorMax']]
        # Determine whether or not to use auto-log scale.
        if plotDict['logScale'] == 'auto':
            if plotDict['colorMin'] > 0:
                if np.log10(plotDict['colorMax'])-np.log10(plotDict['colorMin']) > 3:
                    plotDict['logScale'] = True
                else:
                    plotDict['logScale'] = False
            else:
                plotDict['logScale'] = False
        if plotDict['logScale']:
            # Move min/max values to things that can be marked on the colorbar.
            plotDict['colorMin'] = 10**(int(np.log10(plotDict['colorMin'])))
            plotDict['colorMax'] = 10**(int(np.log10(plotDict['colorMax'])))
        # Add ellipses at RA/Dec locations
        lon = -(slicer.slicePoints['ra'][mask] - plotDict['raCen'] - np.pi) % (np.pi*2) - np.pi
        ellipses = self._plot_tissot_ellipse(lon, slicer.slicePoints['dec'][mask], plotDict['radius'], rasterized=True, ax=ax)
        if plotDict['metricIsColor']:
            current = None
            for ellipse, mVal in zip(ellipses, metricValue.data[mask]):
                if mVal[3] > 1:
                    ellipse.set_alpha(1.0)
                    ellipse.set_facecolor((mVal[0], mVal[1], mVal[2]))
                    ellipse.set_edgecolor('k')
                    current = ellipse
                else:
                    ellipse.set_alpha(mVal[3])
                    ellipse.set_color((mVal[0], mVal[1], mVal[2]))
                ax.add_patch(ellipse)
            if current:
                ax.add_patch(current)
        else:
            if plotDict['logScale']:
                norml = colors.LogNorm()
                p = PatchCollection(ellipses, cmap=plotDict['cmap'], alpha=plotDict['alpha'],
                                    linewidth=0, edgecolor=None, norm=norml, rasterized=True)
            else:
                p = PatchCollection(ellipses, cmap=plotDict['cmap'], alpha=plotDict['alpha'],
                                    linewidth=0, edgecolor=None, rasterized=True)
            p.set_array(metricValue.data[mask])
            p.set_clim(clims)
            ax.add_collection(p)
            # Add color bar (with optional setting of limits)
            if plotDict['cbar']:
                cb = plt.colorbar(p, aspect=25, extend='both', extendrect=True, orientation='horizontal',
                                format=plotDict['cbarFormat'])
                # If outputing to PDF, this fixes the colorbar white stripes
                if plotDict['cbar_edge']:
                    cb.solids.set_edgecolor("face")
                cb.set_label(plotDict['xlabel'])
        # Add ecliptic
        self._plot_ecliptic(plotDict['raCen'], ax=ax)
        if plotDict['mwZone']:
            self._plot_mwZone(plotDict['raCen'], ax=ax)
        ax.grid(True, zorder=1)
        ax.xaxis.set_ticklabels([])
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.75, 0.9, '%s' %plotDict['label'])
        if plotDict['title'] is not None:
            plt.text(0.5, 1.09, plotDict['title'], horizontalalignment='center', transform=ax.transAxes)
        return fig.number


class HealpixSDSSSkyMap(BasePlotter):
    def __init__(self):
        self.plotType = 'SkyMap'
        self.objectPlotter = False
        self.defaultPlotDict = {'title':None, 'xlabel':None, 'logScale':False,
                                'cbarFormat':'%.2f', 'cmap':cm.jet,
                                'percentileClip':None, 'colorMin':None,
                                'colorMax':None, 'zp':None, 'normVal':None,
                                'cbar_edge':True, 'label':None, 'raMin':-90,
                                'raMax':90, 'raLen':45, 'decMin':-2., 'decMax':2.}

    def __call__(self, metricValueIn, slicer, userPlotDict, fignum=None, ):

        """
        Plot the sky map of metricValue using healpy cartview plots in thin strips.
        raMin: Minimum RA to plot (deg)
        raMax: Max RA to plot (deg).  Note raMin/raMax define the centers that will be plotted.
        raLen:  Length of the plotted strips in degrees
        decMin: minimum dec value to plot
        decMax: max dec value to plot
        metricValueIn: metric values
        """

        fig = plt.figure(fignum)
        plotDict = {}
        plotDict.update(self.defaultPlotDict)
        plotDict.update(userPlotDict)

        norm = None
        if plotDict['logScale']:
            norm = 'log'
        if plotDict['cmap'] is None:
            cmap = cm.jet
        else:
            cmap = plotDict['cmap']
        if type(cmap) == str:
            cmap = getattr(cm,cmap)
        # Make colormap compatible with healpy
        cmap = colors.LinearSegmentedColormap('cmap', cmap._segmentdata, cmap.N)
        cmap.set_over(cmap(1.0))
        cmap.set_under('w')
        cmap.set_bad('gray')
        if plotDict['zp']:
            metricValue = metricValueIn - plotDict['zp']
        elif plotDict['normVal']:
            metricValue = metricValueIn/plotDict['normVal']
        else:
            metricValue = metricValueIn

        if plotDict['percentileClip']:
            pcMin, pcMax = percentileClipping(metricValue.compressed(),
                                              percentile=plotDict['percentileClip'])
        if plotDict['colorMin'] is None and plotDict['percentileClip']:
            plotDict['colorMin'] = pcMin
        if plotDict['colorMax'] is None and plotDict['percentileClip']:
            plotDict['colorMax'] = pcMax
        if (plotDict['colorMin'] is not None) or (plotDict['colorMax'] is not None):
            clims = [plotDict['colorMin'], plotDict['colorMax']]
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
        racenters=np.arange(plotDict['raMin'],plotDict['raMax'],plotDict['raLen'])
        nframes = racenters.size
        for i, racenter in enumerate(racenters):
            if i == 0:
                useTitle = plotDict['title'] +' /n'+'%i < RA < %i'%(racenter-plotDict['raLen'], racenter+plotDict['raLen'])
            else:
                useTitle = '%i < RA < %i'%(racenter-plotDict['raLen'], racenter+plotDict['raLen'])
            hp.cartview(metricValue.filled(slicer.badval), title=useTitle, cbar=False,
                        min=clims[0], max=clims[1], flip='astro', rot=(racenter,0,0),
                        cmap=cmap, norm=norm, lonra=[-plotDict['raLen'],plotDict['raLen']],
                        latra=[plotDict['decMin'],plotDict['decMax']], sub=(nframes+1,1,i+1), fig=fig)
            hp.graticule(dpar=20, dmer=20, verbose=False)
        # Add colorbar (not using healpy default colorbar because want more tickmarks).
        fig = plt.gcf()
        ax1 = fig.add_axes([0.1, .15,.8,.075]) #left, bottom, width, height
        # Add label.
        if plotDict['label'] is not None:
            plt.figtext(0.8, 0.9, '%s' %plotDict['label'])
        # Make the colorbar as a seperate figure,
        # from http://matplotlib.org/examples/api/colorbar_only.html
        cnorm = colors.Normalize(vmin=clims[0], vmax=clims[1])
        # supress silly colorbar warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=cnorm,
                                            orientation='horizontal', format=plotDict['cbarFormat'])
            cb1.set_label(plotDict['xlabel'])
        # If outputing to PDF, this fixes the colorbar white stripes
        if plotDict['cbar_edge']:
            cb1.solids.set_edgecolor("face")
        fig = plt.gcf()
        return fig.number
